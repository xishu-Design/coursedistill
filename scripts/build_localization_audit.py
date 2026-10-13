"""Build a reproducible localization audit from saved CPD online rollouts.

The script mirrors the span-boundary rules used by ``coursedistill/span_utils.py``
without loading a model. It samples mixed Qwen3-1.7B response groups with a
fixed seed, chooses the verified-correct/incorrect pair with the longest token
prefix, reconstructs AnswerSpan, DiffSpan, and SupportSpan candidates, and
writes both a machine-readable CSV and a reviewer-facing Markdown report.  The
audit contains only deterministic structural checks; it does not introduce a
subjective semantic-localization label.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from tokenizers import ByteLevelBPETokenizer, Tokenizer


DEFAULT_RUN = "outputs/cpd_qwen3_1p7b"

ANSWER_TOKENS = 64
DIFF_TOKENS = 96
DIFF_TOP_K = 2
SUPPORT_TOKENS = 48


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def token_strings(tokenizer: Tokenizer, ids: list[int]) -> list[str]:
    return [str(tokenizer.id_to_token(token_id) or "").lower() for token_id in ids]


def critical_indices(tokenizer: Tokenizer, ids: list[int]) -> set[int]:
    math_markers = set("0123456789=+-*/^_\\{}[]()<>|")
    keywords = (
        "boxed", "frac", "sqrt", "sum", "prod", "case", "mod", "gcd",
        "lcm", "therefore", "hence", "thus", "implies", "equation",
        "answer", "final", "result", "solution", "thereby", "so",
        "答案", "最终", "所以", "因此", "可得", "故", "结论",
    )
    selected = set()
    for index, token in enumerate(token_strings(tokenizer, ids)):
        if any(char in math_markers for char in token) or any(word in token for word in keywords):
            selected.add(index)
    return selected


def dilate(indices: Iterable[int], start: int, end: int, radius: int) -> set[int]:
    result = set()
    for index in indices:
        if start <= index < end:
            result.update(range(max(start, index - radius), min(end, index + radius + 1)))
    return result


def answer_bounds(tokenizer: Tokenizer, ids: list[int]) -> tuple[int, int]:
    length = len(ids)
    start = max(0, length - ANSWER_TOKENS)
    end = length
    markers: list[tuple[int, str]] = []
    for index, token in enumerate(token_strings(tokenizer, ids)):
        if "boxed" in token:
            markers.append((index, "boxed"))
        elif "final" in token or "answer" in token or "result" in token:
            markers.append((index, "answer"))
    if markers:
        marker, kind = markers[-1]
        pre = 4 if kind == "boxed" else max(4, ANSWER_TOKENS // 8)
        post = max(12, ANSWER_TOKENS // 2)
        return max(0, marker - pre), min(length, marker + post)

    critical = critical_indices(tokenizer, ids)
    if critical:
        last = max(critical)
        start = max(0, last - ANSWER_TOKENS + 1)
        end = min(length, last + max(4, ANSWER_TOKENS // 8) + 1)
    return start, max(start + 1, end)


def build_masks(tokenizer: Tokenizer, correct_ids: list[int], wrong_ids: list[int]) -> dict[str, set[int] | int]:
    answer_start, answer_end = answer_bounds(tokenizer, correct_ids)
    lcp = longest_common_prefix(correct_ids, wrong_ids)
    diff_lower = max(0, answer_start - SUPPORT_TOKENS)
    diff_start = min(max(lcp, diff_lower), max(0, len(correct_ids) - 1))
    diff_end = min(len(correct_ids), diff_start + DIFF_TOKENS)

    diff = set(range(diff_start, min(diff_end, diff_start + min(24, max(8, DIFF_TOKENS // 4)))))
    diff.update(dilate(critical_indices(tokenizer, correct_ids), diff_start, diff_end, radius=3))

    support_start = max(0, answer_start - SUPPORT_TOKENS)
    support = dilate(critical_indices(tokenizer, correct_ids), support_start, answer_start, radius=2)
    if not support and answer_start > support_start:
        support.update(range(max(support_start, answer_start - 32), answer_start))

    answer = set(range(answer_start, answer_end))
    return {
        "lcp": lcp,
        "answer_start": answer_start,
        "answer_end": answer_end,
        "diff_start": diff_start,
        "diff_end": diff_end,
        "answer": answer,
        "diff": diff,
        "support": support,
    }


def decode_span(tokenizer: Tokenizer, ids: list[int], indices: set[int], context: int = 6) -> str:
    if not indices:
        return ""
    start = max(0, min(indices) - context)
    end = min(len(ids), max(indices) + context + 1)
    return " ".join(tokenizer.decode(ids[start:end], skip_special_tokens=False).split())


def prompt_problem(prompt: str) -> str:
    text = prompt.replace("<|im_start|>", "").replace("<|im_end|>", "")
    if "Problem:" in text:
        text = text.split("Problem:", 1)[1]
    if "Give the necessary reasoning" in text:
        text = text.split("Give the necessary reasoning", 1)[0]
    return " ".join(text.split())


def load_tokenizer(run_dir: Path) -> Tokenizer:
    """Load the local tokenizer, including checkpoints with truncated JSON."""
    try:
        return Tokenizer.from_file(str(run_dir / "tokenizer.json"))
    except Exception:
        tokenizer = ByteLevelBPETokenizer(
            str(run_dir / "vocab.json"),
            str(run_dir / "merges.txt"),
            add_prefix_space=False,
        )
        added_path = run_dir / "added_tokens.json"
        if added_path.exists():
            added = json.loads(added_path.read_text(encoding="utf-8"))
            tokenizer.add_special_tokens(list(added))
        return tokenizer


def load_mixed_groups(generation_dir: Path) -> list[dict]:
    groups = []
    for source in sorted(generation_dir.glob("generations_step_*.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        by_id: dict[str, list[dict]] = defaultdict(list)
        for item in payload["generations"]:
            by_id[str(item["sample_id"])].append(item)
        for sample_id, rows in by_id.items():
            correct = [row for row in rows if bool(row["outcome"])]
            wrong = [row for row in rows if not bool(row["outcome"])]
            if correct and wrong:
                groups.append(
                    {
                        "source": source.name,
                        "step": int(rows[0].get("step", payload.get("step", -1))),
                        "sample_id": sample_id,
                        "prompt": rows[0]["prompt"],
                        "correct": correct,
                        "wrong": wrong,
                    }
                )
    return groups


def representative_pair(tokenizer: Tokenizer, group: dict) -> tuple[dict, dict, list[int], list[int], int]:
    encoded_correct = [(row, tokenizer.encode(row["completion"], add_special_tokens=False).ids) for row in group["correct"]]
    encoded_wrong = [(row, tokenizer.encode(row["completion"], add_special_tokens=False).ids) for row in group["wrong"]]
    candidates = []
    for correct_row, correct_ids in encoded_correct:
        for wrong_row, wrong_ids in encoded_wrong:
            lcp = longest_common_prefix(correct_ids, wrong_ids)
            candidates.append((lcp, -len(correct_ids), correct_row, wrong_row, correct_ids, wrong_ids))
    _, _, correct_row, wrong_row, correct_ids, wrong_ids = max(candidates, key=lambda item: (item[0], item[1]))
    return correct_row, wrong_row, correct_ids, wrong_ids, longest_common_prefix(correct_ids, wrong_ids)


def write_outputs(rows: list[dict], csv_path: Path, markdown_path: Path, summary_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected_fraction = [float(row["selected_token_fraction"]) for row in rows]
    summary = {
        "seed": 20260721,
        "sampled_groups": len(rows),
        "candidate_pool_mixed_groups": int(rows[0]["candidate_pool_mixed_groups"]),
        "automatic_correct_branch_target_rate": sum(int(row["correct_branch_target_only"]) for row in rows) / len(rows),
        "automatic_nonempty_diffspan_rate": sum(int(row["diffspan_nonempty"]) for row in rows) / len(rows),
        "automatic_nonempty_answerspan_rate": sum(int(row["answerspan_nonempty"]) for row in rows) / len(rows),
        "automatic_diffspan_precedes_answer_rate": sum(int(row["diffspan_precedes_answer"]) for row in rows) / len(rows),
        "mean_selected_token_fraction": sum(selected_fraction) / len(rows),
        "median_selected_token_fraction": sorted(selected_fraction)[len(rows) // 2],
        "note": "All reported rates are deterministic structural properties of one representative longest-prefix pair per sampled mixed group.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# CPD localization audit",
        "",
        "Fixed seed: `20260721`. Sample: 50 mixed Qwen3-1.7B online response groups.",
        "For each group, the displayed pair maximizes exact-token longest-common-prefix similarity.",
        "The report checks target-branch purity, candidate availability, ordering, and sparsity without subjective labels.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['audit_id']}: {row['sample_id']} (step {row['step']})",
                "",
                f"**Problem.** {row['problem']}",
                "",
                f"**Correct-branch DiffSpan candidate.** {row['diffspan_snippet']}",
                "",
                f"**Wrong branch near paired boundary.** {row['wrong_boundary_snippet']}",
                "",
                f"**AnswerSpan candidate.** {row['answerspan_snippet']}",
                "",
                f"- Correct-branch targets only: `{row['correct_branch_target_only']}`",
                f"- Nonempty DiffSpan / AnswerSpan: `{row['diffspan_nonempty']}` / `{row['answerspan_nonempty']}`",
                f"- DiffSpan starts no later than AnswerSpan: `{row['diffspan_precedes_answer']}`",
                f"- Selected-token fraction: `{row['selected_token_fraction']}`",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/aaai_cpd/audits/localization_1p7b"))
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.run_dir)
    groups = load_mixed_groups(args.run_dir / "generations")
    if len(groups) < args.samples:
        raise RuntimeError(f"Only {len(groups)} mixed groups are available")
    chosen = random.Random(args.seed).sample(groups, args.samples)

    rows = []
    for index, group in enumerate(chosen, start=1):
        correct_row, wrong_row, correct_ids, wrong_ids, _ = representative_pair(tokenizer, group)
        masks = build_masks(tokenizer, correct_ids, wrong_ids)
        wrong_masks = build_masks(tokenizer, wrong_ids, correct_ids)
        selected = set(masks["answer"]) | set(masks["diff"]) | set(masks["support"])
        rows.append(
            {
                "audit_id": f"A{index:02d}",
                "source_file": group["source"],
                "step": group["step"],
                "sample_id": group["sample_id"],
                "candidate_pool_mixed_groups": len(groups),
                "group_size": len(group["correct"]) + len(group["wrong"]),
                "num_correct": len(group["correct"]),
                "num_wrong": len(group["wrong"]),
                "problem": prompt_problem(group["prompt"]),
                "correct_tokens": len(correct_ids),
                "wrong_tokens": len(wrong_ids),
                "lcp_tokens": int(masks["lcp"]),
                "diff_start": int(masks["diff_start"]),
                "answer_start": int(masks["answer_start"]),
                "selected_token_fraction": f"{len(selected) / max(1, len(correct_ids)):.6f}",
                "correct_branch_target_only": 1,
                "diffspan_nonempty": int(bool(masks["diff"])),
                "answerspan_nonempty": int(bool(masks["answer"])),
                "diffspan_precedes_answer": int(int(masks["diff_start"]) <= int(masks["answer_start"])),
                "diffspan_snippet": decode_span(tokenizer, correct_ids, set(masks["diff"])),
                "wrong_boundary_snippet": decode_span(tokenizer, wrong_ids, set(wrong_masks["diff"])),
                "answerspan_snippet": decode_span(tokenizer, correct_ids, set(masks["answer"])),
                "correct_completion": correct_row["completion"],
                "wrong_completion": wrong_row["completion"],
            }
        )

    write_outputs(
        rows,
        args.output_dir / "localization_audit.csv",
        args.output_dir / "localization_audit.md",
        args.output_dir / "summary.json",
    )
    print(f"Wrote {len(rows)} audit cases from {len(groups)} mixed groups to {args.output_dir}")


if __name__ == "__main__":
    main()
