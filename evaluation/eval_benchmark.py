import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from eval_math500 import (
    apply_chat_prompt,
    call_with_timeout,
    generate_batch,
    load_eval_model,
    read_jsonl,
    write_jsonl,
)
from verifier import extract_boxed_answer


_BOX_MARKERS = (r"\boxed{", r"\fbox{")


def _balanced_content(text: str, open_brace: int) -> str | None:
    depth = 0
    start = None
    for idx in range(open_brace, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
            if start is None:
                start = idx + 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:idx]
    return None


def _boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    for marker in _BOX_MARKERS:
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            content = _balanced_content(text, idx + len(marker) - 1)
            if content is not None and content.strip():
                answers.append(content.strip())
            start = idx + len(marker)
    return answers


def _strip_outer_math(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^\\\(|\\\)$", "", text)
    text = re.sub(r"^\\\[|\\\]$", "", text)
    text = text.strip()
    if len(text) >= 2 and text[0] == "$" and text[-1] == "$":
        text = text.strip("$").strip()
    return text


def _remove_latex_noise(text: str) -> str:
    text = _strip_outer_math(text)
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\,": "",
        r"\!": "",
        r"\;": "",
        r"\:": "",
        r"\cdot": "*",
        r"\times": "*",
        r"\dfrac": r"\frac",
        r"\tfrac": r"\frac",
        r"\operatorname": "",
        "−": "-",
        "–": "-",
        "—": "-",
        "，": ",",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\(?:mathrm|text|mbox)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:rm|it|bf)\s+", "", text)
    text = text.replace("~", "")
    text = text.strip()
    return text


def _strip_units_for_minerva(text: str) -> str:
    text = _remove_latex_noise(text)
    text = re.sub(r"\\?mathrm\{[^{}]*\}", "", text)
    text = re.sub(r"\\?text\{[^{}]*\}", "", text)
    text = re.sub(
        r"\b(?:ergs?|seconds?|sec|s|cm|m|km|pc|parsecs?|rad|radians?|degrees?|kelvin|k|kg|hz|joules?|watts?)\b",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bper\b|/", "/", text, flags=re.I)
    return text.strip()


def _split_top_level(text: str) -> list[str]:
    text = _strip_outer_math(text)
    if not text:
        return []
    parts: list[str] = []
    start = 0
    depth = 0
    pairs = {"{": "}", "[": "]", "(": ")"}
    closing = set(pairs.values())
    for idx, char in enumerate(text):
        if char in pairs:
            depth += 1
        elif char in closing and depth > 0:
            depth -= 1
        elif char in {",", ";"} and depth == 0:
            part = text[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    if len(parts) <= 1 and re.search(r"\s+and\s+", text, flags=re.I):
        and_parts = [part.strip() for part in re.split(r"\s+and\s+", text, flags=re.I) if part.strip()]
        if len(and_parts) > 1:
            return and_parts
    return parts or [text]


def _answer_candidates(text: str) -> list[str]:
    text = str(text or "")
    candidates: list[str] = []
    boxed = _boxed_answers(text)
    candidates.extend(boxed)
    if len(boxed) > 1:
        candidates.append(",".join(boxed))

    patterns = [
        r"(?:final answer is|the answer is|answer is|answer:|therefore,?)\s*([^\n]+)",
        r"(?:答案是|答案为|最终答案为|最终答案是)\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        candidates.extend(match.strip().rstrip(".") for match in matches if match.strip())

    latex_matches = re.findall(r"\$([^$]+)\$", text)
    if latex_matches:
        candidates.append(latex_matches[-1].strip())

    fallback = extract_boxed_answer(text)
    if fallback:
        candidates.append(fallback)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidates.append(lines[-1].strip().rstrip("."))

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        candidate = _strip_outer_math(candidate).strip()
        key = candidate.lower()
        if candidate and key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _normalize_for_string(text: str, *, minerva: bool = False) -> str:
    text = _strip_units_for_minerva(text) if minerva else _remove_latex_noise(text)
    text = text.lower()
    text = text.replace(" ", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = text.replace("%", "")
    text = text.replace("^\\circ", "").replace("circ", "")
    text = text.rstrip(".")
    return text


def _latex_frac_to_python(text: str) -> str:
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while True:
        new = pattern.sub(r"((\1)/(\2))", text)
        if new == text:
            return new
        text = new


def _latex_sci_to_python(text: str) -> str:
    text = re.sub(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:\\times|\*|x)\s*10\s*\^\s*\{?([+-]?\d+)\}?",
        r"\1e\2",
        text,
        flags=re.I,
    )
    return text


def _insert_implicit_multiplication(text: str) -> str:
    text = re.sub(r"(\d|\))\s*(\()", r"\1*\2", text)
    text = re.sub(r"(\d)\s*(pi|sqrt)\b", r"\1*\2", text)
    text = re.sub(r"(\))\s*(\d|pi|sqrt)\b", r"\1*\2", text)
    return text


def _to_float(text: str, *, minerva: bool = False) -> float | None:
    text = _strip_units_for_minerva(text) if minerva else _remove_latex_noise(text)
    text = _latex_frac_to_python(_latex_sci_to_python(text))
    text = text.replace("^", "**")
    text = text.replace(",", "")
    text = text.replace("%", "/100")
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\pi\b", "pi", text)
    text = _insert_implicit_multiplication(text)
    text = text.strip()
    if not text:
        return None

    # If the model appends units or prose, keep the last plausible numeric expression.
    numeric_match = re.findall(
        r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|(?:\(\s*[+-]?\d+(?:\.\d*)?\s*/\s*[+-]?\d+(?:\.\d*)?\s*\)))",
        text,
        flags=re.I,
    )
    candidates = [text]
    if numeric_match:
        candidates.extend(reversed(numeric_match))

    for candidate in candidates:
        try:
            allowed = {"sqrt": math.sqrt, "pi": math.pi, "e": math.e}
            value = eval(candidate, {"__builtins__": {}}, allowed)  # noqa: S307 - restricted numeric eval
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        except Exception:
            pass
        try:
            return float(candidate)
        except Exception:
            pass
    return None


def _parse_sympy(text: str):
    text = _remove_latex_noise(text)
    if not text:
        return None
    try:
        import sympy as sp
        from sympy.parsing.latex import parse_latex

        return parse_latex(_strip_outer_math(text))
    except Exception:
        pass

    try:
        import sympy as sp

        expr = _latex_frac_to_python(_latex_sci_to_python(text))
        expr = expr.replace("^", "**")
        expr = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expr)
        expr = expr.replace("\\pi", "pi")
        expr = expr.replace("\\", "")
        expr = _insert_implicit_multiplication(expr)
        try:
            from sympy.parsing.sympy_parser import (
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )

            transformations = standard_transformations + (implicit_multiplication_application,)
            return parse_expr(expr, transformations=transformations)
        except Exception:
            return sp.sympify(expr)
    except Exception:
        return None


def _sympy_equal(prediction: str, gold: str) -> bool:
    try:
        import sympy as sp

        if "=" in prediction and "=" in gold:
            pred_l, pred_r = prediction.split("=", 1)
            gold_l, gold_r = gold.split("=", 1)
            pred_expr = _parse_sympy(f"({pred_l})-({pred_r})")
            gold_expr = _parse_sympy(f"({gold_l})-({gold_r})")
        else:
            pred_expr = _parse_sympy(prediction)
            gold_expr = _parse_sympy(gold)
        if pred_expr is None or gold_expr is None:
            return False
        diff = sp.simplify(pred_expr - gold_expr)
        return bool(diff == 0)
    except Exception:
        return False


MINERVA_INVALID_ANSWER = "[invalidanswer]"

MINERVA_SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

MINERVA_REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def minerva_last_boxed_only_string(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def minerva_remove_boxed(text: str | None) -> str:
    if not text:
        return MINERVA_INVALID_ANSWER
    if "\\boxed " in text:
        left = "\\boxed "
        if text.startswith(left):
            return text[len(left) :]
        return text
    left = "\\boxed{"
    if text.startswith(left) and text.endswith("}"):
        return text[len(left) : -1]
    if text.startswith("\\fbox{") and text.endswith("}"):
        return text[len("\\fbox{") : -1]
    return text


def minerva_get_unnormalized_answer(text: str) -> str:
    end_seq = "I hope it is correct."
    candidate = text + end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?).\nI hope it is correct.",
        candidate,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    boxed = minerva_last_boxed_only_string(text)
    if boxed is not None:
        return minerva_remove_boxed(boxed).strip()
    generic = _answer_candidates(text)
    if generic:
        return generic[0]
    return MINERVA_INVALID_ANSWER


def minerva_normalize_final_answer(final_answer: str) -> str:
    """Normalize final answer following lm-evaluation-harness Minerva utils."""
    final_answer = str(final_answer or "")
    final_answer = final_answer.split("=")[-1]
    for before, after in MINERVA_SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in MINERVA_REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


def _looks_like_scalar_number(text: str) -> bool:
    text = _strip_units_for_minerva(text)
    text = _latex_frac_to_python(_latex_sci_to_python(text))
    text = text.replace("^", "**")
    text = text.replace(",", "")
    text = text.replace("%", "/100")
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\pi\b", "pi", text)
    text = _insert_implicit_multiplication(text)
    text = text.strip()
    if not text:
        return False
    if re.search(r"[a-df-zA-DF-Z]", text.replace("sqrt", "").replace("pi", "")):
        return False
    return bool(re.fullmatch(r"[0-9eE+\-*/().\s]+", text))


def _scalar_value_from_labeled_prediction(text: str) -> float | None:
    """Accept a scalar answer with harmless labels/units, but not formulas."""
    text = _strip_units_for_minerva(text)
    text = text.replace("\\approx", "=").replace("≈", "=")
    if "=" in text:
        text = text.split("=")[-1]
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\\?frac|[A-Za-z_][A-Za-z0-9_]*\s*\^", text):
        return None
    return _to_float(text, minerva=True)


def minerva_is_equiv(prediction: str, reference: str, rel_tol: float = 2e-2, abs_tol: float = 1e-6) -> bool:
    prediction = minerva_normalize_final_answer(prediction)
    reference = minerva_normalize_final_answer(reference)
    if prediction == reference:
        return True

    pred_scalar = _looks_like_scalar_number(prediction)
    ref_scalar = _looks_like_scalar_number(reference)

    if ref_scalar:
        pred_num = _to_float(prediction, minerva=True) if pred_scalar else _scalar_value_from_labeled_prediction(prediction)
        ref_num = _to_float(reference, minerva=True)
        if pred_num is not None and ref_num is not None:
            return math.isclose(pred_num, ref_num, rel_tol=rel_tol, abs_tol=abs_tol)
        return False
    if pred_scalar:
        return False

    if _sympy_equal(prediction, reference):
        return True
    return False


class OlympiadBenchAutoScoringJudge:
    """Deterministic OpenBMB/OlympiadBench-style scorer.

    The logic follows the public AutoScoringJudge design: extract boxed answers,
    normalize LaTeX noise, split top-level multi-answer strings, then try exact,
    numeric-tolerance, expression, and equation equivalence.
    """

    def __init__(self, precision: float = 1e-6):
        self.precision = precision

    @staticmethod
    def get_boxed_str(text: str) -> str:
        boxed = _boxed_answers(text)
        return ",".join(boxed) if boxed else str(text or "")

    @staticmethod
    def _format_special_signal(text: str) -> str:
        replacements = {
            "\\left": "",
            "\\right": "",
            "\\quad": "",
            "\\qquad": "",
            "\\,": "",
            "\\!": "",
            "\\;": "",
            "\\:": "",
            "\\cdot": "*",
            "\\times": "*",
            "\\dfrac": "\\frac",
            "\\tfrac": "\\frac",
            "\\leq": "<=",
            "\\geq": ">=",
            "\\ne": "!=",
            "\\neq": "!=",
            "\\approx": "=",
            "\\sim": "=",
            "−": "-",
            "–": "-",
            "—": "-",
            "，": ",",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"\\(?:mathrm|operatorname|text|mbox)\{([^{}]*)\}", r"\1", text)
        text = text.replace("\\%", "%")
        return text

    def preprocess(self, text: str) -> str:
        text = self.get_boxed_str(text)
        text = _strip_outer_math(text)
        text = self._format_special_signal(text)
        text = text.strip()
        text = text.rstrip(".")
        return text

    @staticmethod
    def split_by_comma(text: str) -> list[str]:
        return _split_top_level(text)

    @staticmethod
    def trans_plus_minus_sign(text: str) -> list[str]:
        if "\\pm" not in text and "±" not in text:
            return [text]
        plus = text.replace("\\pm", "+").replace("±", "+")
        minus = text.replace("\\pm", "-").replace("±", "-")
        return [plus, minus]

    def numerical_equal(self, prediction: str, reference: str) -> bool:
        pred = _to_float(prediction)
        ref = _to_float(reference)
        if pred is None or ref is None:
            return False
        candidates = [pred]
        if "%" in prediction and "%" not in reference:
            candidates.append(pred / 100.0)
        if "%" in reference and "%" not in prediction:
            candidates.append(pred * 100.0)
        return any(abs(candidate - ref) <= self.precision for candidate in candidates)

    def expression_equal(self, prediction: str, reference: str) -> bool:
        if _sympy_equal(prediction, reference):
            return True
        try:
            import sympy as sp

            pred_expr = _parse_sympy(prediction)
            ref_expr = _parse_sympy(reference)
            if pred_expr is None or ref_expr is None:
                return False
            pred_num = sp.N(pred_expr)
            ref_num = sp.N(ref_expr)
            return bool(abs(float(pred_num) - float(ref_num)) <= self.precision)
        except Exception:
            return False

    def equation_equal(self, prediction: str, reference: str) -> bool:
        if "=" not in prediction or "=" not in reference:
            return False
        pred_left, pred_right = prediction.split("=", 1)
        ref_left, ref_right = reference.split("=", 1)
        return self.expression_equal(f"({pred_left})-({pred_right})", f"({ref_left})-({ref_right})")

    def item_equal(self, prediction: str, reference: str) -> bool:
        prediction = self.preprocess(prediction)
        reference = self.preprocess(reference)
        if _normalize_for_string(prediction) == _normalize_for_string(reference):
            return True
        if self.numerical_equal(prediction, reference):
            return True
        if self.equation_equal(prediction, reference):
            return True
        if self.expression_equal(prediction, reference):
            return True
        return False

    def judge(self, reference: str, prediction: str) -> bool:
        ref = self.preprocess(reference)
        pred = self.preprocess(prediction)
        ref_parts = self.split_by_comma(ref)
        pred_parts = self.split_by_comma(pred)
        if len(ref_parts) != len(pred_parts):
            ref_parts = [ref]
            pred_parts = [pred]
        unused = list(pred_parts)
        for ref_item in ref_parts:
            matched = None
            ref_variants = self.trans_plus_minus_sign(ref_item)
            for idx, pred_item in enumerate(unused):
                pred_variants = self.trans_plus_minus_sign(pred_item)
                if any(self.item_equal(pred_v, ref_v) for pred_v in pred_variants for ref_v in ref_variants):
                    matched = idx
                    break
            if matched is None:
                return False
            unused.pop(matched)
        return True


def _item_equal(
    prediction: str,
    gold: str,
    *,
    benchmark: str,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    minerva = benchmark == "minerval"
    pred_norm = _normalize_for_string(prediction, minerva=minerva)
    gold_norm = _normalize_for_string(gold, minerva=minerva)
    if pred_norm and pred_norm == gold_norm:
        return True

    pred_num = _to_float(prediction, minerva=minerva)
    gold_num = _to_float(gold, minerva=minerva)
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=rel_tol, abs_tol=abs_tol)

    if _sympy_equal(prediction, gold):
        return True

    return False


class BenchmarkGrader:
    def __init__(
        self,
        benchmark: str,
        rel_tol: float | None = None,
        abs_tol: float | None = None,
        olympiadbench_eval_dir: str | None = None,
        prefer_official_olympiadbench: bool = False,
    ):
        self.benchmark = benchmark
        if rel_tol is None:
            rel_tol = 2e-2 if benchmark == "minerval" else 1e-6
        if abs_tol is None:
            abs_tol = 1e-30 if benchmark == "minerval" else 1e-6
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.official_olympiadbench_judge = None
        if benchmark == "minerval":
            self.grader_backend = "minerva_boxed_numeric_sympy"
        elif benchmark == "olympiadbench":
            self.grader_backend = "builtin_olympiadbench_auto_scoring_judge"
            self.official_olympiadbench_judge = OlympiadBenchAutoScoringJudge(precision=self.abs_tol)
            if prefer_official_olympiadbench:
                external_judge = load_official_olympiadbench_judge(olympiadbench_eval_dir)
                if external_judge is not None:
                    self.official_olympiadbench_judge = external_judge
                    self.grader_backend = "external_olympiadbench_auto_scoring_judge"
        else:
            self.grader_backend = "local_compatible"

    def grade(self, response: str, row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        gold = str(row["answer"])
        if self.benchmark == "minerval":
            prediction = minerva_get_unnormalized_answer(response)
            correct = minerva_is_equiv(prediction, gold, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            return correct, {
                "prediction": prediction,
                "gold_candidates": [gold],
                "prediction_candidates": [prediction],
                "grader_backend": self.grader_backend,
            }

        pred_candidates = _answer_candidates(response)
        gold_candidates = _answer_candidates(gold) or [gold]

        if self.official_olympiadbench_judge is not None:
            for pred in pred_candidates:
                try:
                    try:
                        accepted = self.official_olympiadbench_judge.judge(gold, pred)
                    except TypeError:
                        accepted = self.official_olympiadbench_judge.judge(gold, pred, self.abs_tol)
                    if bool(accepted):
                        return True, {
                            "prediction": pred,
                            "gold_candidates": gold_candidates,
                            "prediction_candidates": pred_candidates,
                            "grader_backend": self.grader_backend,
                        }
                except Exception:
                    pass
            return False, {
                "prediction": pred_candidates[0] if pred_candidates else "",
                "gold_candidates": gold_candidates,
                "prediction_candidates": pred_candidates,
                "grader_backend": self.grader_backend,
            }

        for pred in pred_candidates:
            if self._candidate_matches(pred, gold_candidates, row):
                return True, {
                    "prediction": pred,
                    "gold_candidates": gold_candidates,
                    "prediction_candidates": pred_candidates,
                    "grader_backend": self.grader_backend,
                }
        return False, {
            "prediction": pred_candidates[0] if pred_candidates else "",
            "gold_candidates": gold_candidates,
            "prediction_candidates": pred_candidates,
            "grader_backend": self.grader_backend,
        }

    def _candidate_matches(self, prediction: str, gold_candidates: list[str], row: dict[str, Any]) -> bool:
        pred_parts = _split_top_level(prediction)
        for gold in gold_candidates:
            gold_parts = _split_top_level(gold)
            if self.benchmark == "olympiadbench" and (row.get("is_multiple_answer") or len(gold_parts) > 1):
                if len(pred_parts) != len(gold_parts):
                    continue
                unused = list(pred_parts)
                all_match = True
                for gold_part in gold_parts:
                    hit_index = None
                    for idx, pred_part in enumerate(unused):
                        if _item_equal(
                            pred_part,
                            gold_part,
                            benchmark=self.benchmark,
                            rel_tol=self.rel_tol,
                            abs_tol=self.abs_tol,
                        ):
                            hit_index = idx
                            break
                    if hit_index is None:
                        all_match = False
                        break
                    unused.pop(hit_index)
                if all_match:
                    return True
            else:
                if _item_equal(
                    prediction,
                    gold,
                    benchmark=self.benchmark,
                    rel_tol=self.rel_tol,
                    abs_tol=self.abs_tol,
                ):
                    return True
        return False


def load_official_olympiadbench_judge(eval_dir: str | None):
    candidates: list[Path] = []
    if eval_dir:
        candidates.append(Path(eval_dir))
    env_dir = os.environ.get("OLYMPIADBENCH_EVAL_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path("third_party/OlympiadBench/eval"),
            Path("third_party/OlympiadBench"),
        ]
    )

    for candidate in candidates:
        eval_path = candidate / "eval" if (candidate / "eval" / "auto_scoring_judge.py").exists() else candidate
        judge_file = eval_path / "auto_scoring_judge.py"
        if not judge_file.exists():
            continue
        sys.path.insert(0, str(eval_path.resolve()))
        try:
            from auto_scoring_judge import AutoScoringJudge

            return AutoScoringJudge()
        except Exception as exc:
            print(f"Warning: failed to load official OlympiadBench judge from {judge_file}: {exc}", file=sys.stderr)
        finally:
            try:
                sys.path.remove(str(eval_path.resolve()))
            except ValueError:
                pass
    print(
        "Warning: official OlympiadBench AutoScoringJudge not found; "
        "using local compatible grader. Set OLYMPIADBENCH_EVAL_DIR=/path/to/OlympiadBench/eval "
        "for official scoring.",
        file=sys.stderr,
    )
    return None


def summarize(rows: list[dict[str, Any]], num_samples: int) -> dict[str, Any]:
    total = len({row["unique_id"] for row in rows})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["unique_id"]].append(row)

    avg_correct = []
    first_correct = []
    pass_correct = []
    by_subject = defaultdict(lambda: {"count": 0, "avg": 0.0, "first": 0, "passed": 0})
    by_answer_type = defaultdict(lambda: {"count": 0, "avg": 0.0, "first": 0, "passed": 0})

    for _, samples in grouped.items():
        samples = sorted(samples, key=lambda item: item["sample_index"])
        correctness = [bool(sample["correct"]) for sample in samples]
        avg = sum(correctness) / max(1, len(correctness))
        first = correctness[0] if correctness else False
        passed = any(correctness)
        avg_correct.append(avg)
        first_correct.append(first)
        pass_correct.append(passed)

        subject = samples[0].get("subject", "")
        answer_type = samples[0].get("answer_type", "")
        for bucket, key in ((by_subject, subject), (by_answer_type, answer_type)):
            bucket[key]["count"] += 1
            bucket[key]["avg"] += avg
            bucket[key]["first"] += int(first)
            bucket[key]["passed"] += int(passed)

    def finalize(bucket):
        output = {}
        for key, value in sorted(bucket.items()):
            count = max(1, value["count"])
            output[str(key)] = {
                "count": value["count"],
                f"avg_at_{num_samples}": value["avg"] / count,
                f"pass_at_{num_samples}": value["passed"] / count,
                f"first_at_{num_samples}": value["first"] / count,
            }
        return output

    return {
        "count": total,
        "num_samples": num_samples,
        f"avg_at_{num_samples}": sum(avg_correct) / max(1, total),
        f"pass_at_{num_samples}": sum(pass_correct) / max(1, total),
        f"first_at_{num_samples}": sum(first_correct) / max(1, total),
        "by_subject": finalize(by_subject),
        "by_answer_type": finalize(by_answer_type),
    }


def infer_benchmark(path: Path, rows: list[dict[str, Any]]) -> str:
    text = str(path).lower()
    if "minerva" in text or "minerval" in text:
        return "minerval"
    if "olymp" in text:
        return "olympiadbench"
    sources = {str(row.get("source_dataset", "")).lower() for row in rows[:20]}
    if any("minerva" in source for source in sources):
        return "minerval"
    if any("olymp" in source for source in sources):
        return "olympiadbench"
    raise ValueError("Could not infer benchmark; pass --benchmark minerval or --benchmark olympiadbench.")


def evaluate_existing_generations(args, grader: BenchmarkGrader, test_rows: list[dict[str, Any]], run_dir: Path) -> None:
    rows_by_id = {str(row["unique_id"]): row for row in test_rows}
    generations = read_jsonl(Path(args.generations_file))
    outputs = []
    for item in generations:
        row = rows_by_id.get(str(item.get("unique_id")), item)
        correct, details = grader.grade(str(item.get("completion", "")), row)
        item = dict(item)
        item.update(
            {
                "correct": correct,
                "grader_prediction": details["prediction"],
                "grader_gold_candidates": details["gold_candidates"],
                "grader_prediction_candidates": details["prediction_candidates"],
                "grader_backend": details.get("grader_backend", grader.grader_backend),
            }
        )
        outputs.append(item)
    write_jsonl(run_dir / "regraded_generations.jsonl", outputs)
    metrics = summarize(outputs, args.num_samples)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Regrade complete: {run_dir}")
    print(f"Avg@{args.num_samples}: {metrics[f'avg_at_{args.num_samples}']:.4f}")
    print(f"Pass@{args.num_samples}: {metrics[f'pass_at_{args.num_samples}']:.4f}")
    print(f"First@{args.num_samples}: {metrics[f'first_at_{args.num_samples}']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Minerva-Math or OlympiadBench-style jsonl files.")
    parser.add_argument("--model_name_or_path", required=False)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--benchmark", choices=["auto", "minerval", "olympiadbench"], default="auto")
    parser.add_argument("--output_dir", default="outputs/benchmark_eval")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--student_thinking", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_lora_rank", type=int, default=64)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=0)
    parser.add_argument("--vllm_max_model_len", type=int, default=0)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument(
        "--sequential_samples",
        action="store_true",
        help="Generate num_samples one round at a time to reduce peak KV-cache memory.",
    )
    parser.add_argument("--log_grading", action="store_true", help="Print per-sample grading progress.")
    parser.add_argument("--grade_timeout_sec", type=float, default=0.0, help="Mark a sample wrong if grading exceeds this many seconds.")
    parser.add_argument("--save_every", type=int, default=0, help="Write partial generations/metrics every N finished batches.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rel_tol", type=float, default=None)
    parser.add_argument("--abs_tol", type=float, default=None)
    parser.add_argument(
        "--olympiadbench_eval_dir",
        default=None,
        help="Path to OpenBMB/OlympiadBench/eval; uses official AutoScoringJudge when available.",
    )
    parser.add_argument(
        "--use_official_olympiadbench_judge",
        action="store_true",
        help="Opt in to OpenBMB/OlympiadBench AutoScoringJudge if its eval directory is available.",
    )
    parser.add_argument("--generations_file", default=None, help="Only regrade an existing generations.jsonl file.")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    test_rows = read_jsonl(test_path)
    if args.limit > 0:
        test_rows = test_rows[: args.limit]
    missing = [field for field in ("problem", "answer", "unique_id") if any(field not in row for row in test_rows)]
    if missing:
        raise ValueError(f"{args.test_file} is missing required fields: {sorted(set(missing))}")

    benchmark = infer_benchmark(test_path, test_rows) if args.benchmark == "auto" else args.benchmark
    grader = BenchmarkGrader(
        benchmark,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        olympiadbench_eval_dir=args.olympiadbench_eval_dir,
        prefer_official_olympiadbench=args.use_official_olympiadbench_judge,
    )

    run_name = args.run_name
    if not run_name:
        model_label = Path(args.adapter_path or args.model_name_or_path or "regrade").name
        run_name = f"{model_label}_{benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.generations_file:
        evaluate_existing_generations(args, grader, test_rows, run_dir)
        return

    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required unless --generations_file is used.")

    import torch

    torch.manual_seed(args.seed)
    tokenizer, model = load_eval_model(args)
    outputs = []

    for start in range(0, len(test_rows), args.batch_size):
        batch = test_rows[start : start + args.batch_size]
        batch_end = min(start + len(batch), len(test_rows))
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"start batch {start + 1}-{batch_end}/{len(test_rows)}",
            flush=True,
        )
        prompts = [apply_chat_prompt(tokenizer, row["problem"], args.student_thinking) for row in batch]
        completions = generate_batch(tokenizer, model, prompts, args)
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"generated batch {start + 1}-{batch_end}/{len(test_rows)}",
            flush=True,
        )

        for row_index, row in enumerate(batch):
            for sample_index in range(args.num_samples):
                completion = completions[row_index * args.num_samples + sample_index]
                if args.log_grading:
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"grade row {start + row_index + 1}/{len(test_rows)} sample {sample_index + 1}/{args.num_samples}",
                        flush=True,
                    )
                try:
                    correct, details = call_with_timeout(args.grade_timeout_sec, grader.grade, completion, row)
                except TimeoutError as exc:
                    prediction = extract_boxed_answer(completion)
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"grade timeout row {start + row_index + 1}/{len(test_rows)} "
                        f"sample {sample_index + 1}/{args.num_samples}: {exc}",
                        flush=True,
                    )
                    correct = False
                    details = {
                        "prediction": prediction,
                        "gold_candidates": [str(row.get("answer", ""))],
                        "prediction_candidates": [prediction] if prediction else [],
                        "grader_backend": f"{grader.grader_backend}_timeout",
                    }
                outputs.append(
                    {
                        "unique_id": row["unique_id"],
                        "subject": row.get("subject", ""),
                        "level": row.get("level", ""),
                        "answer_type": row.get("answer_type", ""),
                        "problem": row["problem"],
                        "answer": row["answer"],
                        "sample_index": sample_index,
                        "prompt": prompts[row_index],
                        "completion": completion,
                        "extracted_answer": extract_boxed_answer(completion),
                        "grader_prediction": details["prediction"],
                        "grader_gold_candidates": details["gold_candidates"],
                        "grader_prediction_candidates": details["prediction_candidates"],
                        "grader_backend": details.get("grader_backend", grader.grader_backend),
                        "correct": correct,
                    }
                )

        done = min(start + len(batch), len(test_rows))
        running = summarize(outputs, args.num_samples)
        print(
            f"[{done}/{len(test_rows)}] avg@{args.num_samples}={running[f'avg_at_{args.num_samples}']:.4f} "
            f"pass@{args.num_samples}={running[f'pass_at_{args.num_samples}']:.4f}",
            flush=True,
        )
        batch_number = start // args.batch_size + 1
        if args.save_every > 0 and batch_number % args.save_every == 0:
            write_jsonl(run_dir / "generations.partial.jsonl", outputs)
            (run_dir / "metrics.partial.json").write_text(
                json.dumps(running, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] wrote partial outputs to {run_dir}",
                flush=True,
            )

    metrics = summarize(outputs, args.num_samples)
    metrics.update(
        {
            "benchmark": benchmark,
            "model_name_or_path": args.model_name_or_path,
            "adapter_path": args.adapter_path,
            "test_file": args.test_file,
            "max_new_tokens": args.max_new_tokens,
            "student_thinking": args.student_thinking,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "rel_tol": grader.rel_tol,
            "abs_tol": grader.abs_tol,
            "grader_backend": grader.grader_backend,
        }
    )
    write_jsonl(run_dir / "generations.jsonl", outputs)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{benchmark} evaluation complete", flush=True)
    print(f"Output: {run_dir}", flush=True)
    print(f"Avg@{args.num_samples}: {metrics[f'avg_at_{args.num_samples}']:.4f}", flush=True)
    print(f"Pass@{args.num_samples}: {metrics[f'pass_at_{args.num_samples}']:.4f}", flush=True)
    print(f"First@{args.num_samples}: {metrics[f'first_at_{args.num_samples}']:.4f}", flush=True)


if __name__ == "__main__":
    main()
