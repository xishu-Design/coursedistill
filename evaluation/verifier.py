import re
from fractions import Fraction
from typing import Optional


def _balanced_brace_content(text: str, start: int) -> Optional[str]:
    depth = 0
    content_start = None
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
            if content_start is None:
                content_start = pos + 1
        elif char == "}":
            depth -= 1
            if depth == 0 and content_start is not None:
                return text[content_start:pos]
    return None


def extract_boxed_answer(text: str) -> str:
    """Extract the last LaTeX boxed answer, falling back to the final nonempty line."""
    if not text:
        return ""

    boxed_markers = [r"\boxed{", r"\fbox{"]
    answers: list[str] = []
    for marker in boxed_markers:
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx == -1:
                break
            content = _balanced_brace_content(text, idx + len(marker) - 1)
            if content is not None and content.strip():
                answers.append(content)
            start = idx + len(marker)

    if answers:
        return answers[-1].strip()

    unbraced_boxed = re.findall(r"\\boxed\s+([^$.,;\n]+)", text)
    if unbraced_boxed:
        return unbraced_boxed[-1].strip()

    if r"\boxed{}" in text and re.search(
        r"\b(no|zero)\s+primes?\b|not prime for any|all .* composite|\\boxed\{\}\s*primes?",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return "0"

    matches = re.findall(
        r"(?:final answer is|answer is|answer:)\s*([$\\{}A-Za-z0-9_./,+\-^= ]+)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def _strip_outer_braces(answer: str) -> str:
    while answer.startswith("{") and answer.endswith("}"):
        depth = 0
        balanced_at_end = False
        for idx, char in enumerate(answer):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    balanced_at_end = idx == len(answer) - 1
                    break
        if not balanced_at_end:
            break
        answer = answer[1:-1].strip()
    return answer


def _simple_numeric_value(answer: str) -> Optional[Fraction]:
    """Parse only unambiguous scalar numeric answers."""
    simple_frac = re.fullmatch(r"\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", answer)
    if simple_frac:
        denom = int(simple_frac.group(2))
        if denom == 0:
            return None
        return Fraction(int(simple_frac.group(1)), denom)

    slash_frac = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", answer)
    if slash_frac:
        denom = int(slash_frac.group(2))
        if denom == 0:
            return None
        return Fraction(int(slash_frac.group(1)), denom)

    if re.fullmatch(r"[+-]?(?:\d+|\d+\.\d*|\.\d+)", answer):
        try:
            return Fraction(answer)
        except ValueError:
            return None
    return None


def _strip_degree_unit(answer: str) -> str:
    """Remove only explicit degree units from otherwise numeric scalar answers."""
    answer = answer.replace("°", "")
    answer = re.sub(r"\^\{?\\+circ\}?", "", answer)
    answer = re.sub(r"\\+(?:degree|degrees)\b", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\\+text\{(?:degree|degrees|deg)\}", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"(?:degrees?|deg)\b", "", answer, flags=re.IGNORECASE)
    return answer


def normalize_answer(answer: str) -> str:
    """Conservative exact-match normalization for math answers."""
    answer = extract_boxed_answer(answer)
    answer = answer.strip()
    answer = re.sub(r"^\$+|\$+$", "", answer)
    answer = answer.strip()
    answer = _strip_outer_braces(answer)
    answer = answer.replace("\u2212", "-")
    answer = re.sub(r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)\b", "", answer)
    answer = re.sub(r"\\(?:dfrac|tfrac|cfrac)\b", r"\\frac", answer)
    answer = answer.replace(r"\left", "").replace(r"\right", "")
    answer = answer.replace(r"\,", "").replace(r"\!", "").replace(r"\;", "").replace(r"\:", "")
    answer = re.sub(r"\\(?:mathrm|operatorname)\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\s+", "", answer)
    answer = answer.rstrip(".")
    answer = answer.replace(",", "")
    answer = _strip_outer_braces(answer)
    return answer


def _answer_equivalence_candidates(answer: str) -> list[str]:
    """Conservative variants that avoid low-estimating final equality answers."""
    candidates = [answer]
    if "=" in answer:
        rhs = answer.rsplit("=", 1)[-1].strip()
        rhs = rhs.strip("$")
        rhs = rhs.strip("{}()[]")
        rhs = rhs.rstrip(".,;:")
        if rhs:
            candidates.append(_strip_outer_braces(rhs))

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def exact_match(response: str, ground_truth: str) -> bool:
    normalized_response = normalize_answer(response)
    normalized_ground_truth = normalize_answer(ground_truth)
    response_candidates = _answer_equivalence_candidates(normalized_response)
    ground_truth_candidates = _answer_equivalence_candidates(normalized_ground_truth)
    if any(response == truth for response in response_candidates for truth in ground_truth_candidates):
        return True

    for response_candidate in response_candidates:
        response_value = _simple_numeric_value(response_candidate)
        if response_value is None:
            continue
        for ground_truth_candidate in ground_truth_candidates:
            ground_truth_value = _simple_numeric_value(ground_truth_candidate)
            if ground_truth_value is not None and response_value == ground_truth_value:
                return True

    for response_candidate in response_candidates:
        response_degree_value = _simple_numeric_value(_strip_degree_unit(response_candidate))
        if response_degree_value is None:
            continue
        for ground_truth_candidate in ground_truth_candidates:
            ground_truth_degree_value = _simple_numeric_value(_strip_degree_unit(ground_truth_candidate))
            if ground_truth_degree_value is not None and response_degree_value == ground_truth_degree_value:
                return True
    return False


def verify_response(response: str, ground_truth: str, verifier_type: str = "exact_match") -> bool:
    verifier_type = (verifier_type or "exact_match").lower()
    if verifier_type != "exact_match":
        raise ValueError(f"Unsupported verifier_type: {verifier_type}")
    return exact_match(response, ground_truth)
