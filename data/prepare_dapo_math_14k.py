from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


OFFICIAL_URL = (
    "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/"
    "resolve/main/data/dapo-math-17k.parquet?download=true"
)
EXPECTED_EOPD_ROWS = 13_904


def repair_latex_control_escapes(text: str) -> str:
    r"""Repair JSON-decoded LaTeX escapes such as ``\f`` in ``\frac``.

    Some source mirrors store LaTeX with a single JSON backslash. After JSON
    decoding, valid JSON escapes like ``\f`` and ``\t`` become control
    characters, e.g. ``\frac`` becomes form-feed + ``rac``. We repair the
    common LaTeX cases instead of silently training on corrupted prompts.
    """
    generic = {
        "\x07": r"\a",
        "\x08": r"\b",
        "\x09": r"\t",
        "\x0b": r"\v",
        "\x0c": r"\f",
        "\x0d": r"\r",
    }
    for bad, good in generic.items():
        text = text.replace(bad, good)

    replacements = {
        r"\frac": r"\frac",
        r"\begin": r"\begin",
        r"\beta": r"\beta",
        r"\times": r"\times",
        r"\text": r"\text",
        r"\theta": r"\theta",
        r"\tan": r"\tan",
        r"\to": r"\to",
        r"\rangle": r"\rangle",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = "".join(char if ord(char) >= 32 or char == "\n" else "" for char in text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official DAPO-Math-17k and convert its English subset for CourseDistill."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-url", default=OFFICIAL_URL)
    parser.add_argument("--source-file", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional deterministic cap after filtering. 0 keeps the full English subset.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, force: bool) -> None:
    if target.exists() and target.stat().st_size > 0 and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    print(f"Downloading official DAPO source:\n  {url}\n  -> {target}")
    urllib.request.urlretrieve(url, partial)
    partial.replace(target)


def load_rows(source_file: Path) -> list[dict]:
    if source_file.suffix.lower() in {".json", ".jsonl"}:
        with source_file.open(encoding="utf-8") as handle:
            if source_file.suffix.lower() == ".jsonl":
                return [json.loads(line) for line in handle if line.strip()]
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list in {source_file}")
        return rows

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The conversion requires datasets/pyarrow. Install with: "
            "python -m pip install -U datasets pyarrow"
        ) from exc
    dataset = load_dataset("parquet", data_files=str(source_file), split="train")
    return list(dataset)


def prompt_from_row(row: dict) -> str:
    prompt = row.get("prompt", "")
    if isinstance(prompt, list):
        user_messages = [
            str(message.get("content", ""))
            for message in prompt
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        prompt = user_messages[-1] if user_messages else ""
    prompt = repair_latex_control_escapes(str(prompt)).strip()
    suffix = "Please reason step by step, and put your final answer within \\boxed{}."
    if prompt.endswith(suffix):
        prompt = prompt[: -len(suffix)].rstrip()
    return prompt


def answer_from_row(row: dict) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return repair_latex_control_escapes(str(reward_model["ground_truth"])).strip()
    for key in ("answer", "ground_truth"):
        if row.get(key) is not None:
            return repair_latex_control_escapes(str(row[key])).strip()
    return ""


def source_id(row: dict, fallback_index: int) -> str:
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("index") is not None:
        return str(extra_info["index"])
    return str(row.get("id") or f"DAPO-{fallback_index}")


def convert(rows: list[dict], max_samples: int) -> tuple[list[dict], dict]:
    output: list[dict] = []
    stats = {
        "source_rows": len(rows),
        "empty_problem_removed": 0,
        "empty_answer_removed": 0,
        "non_ascii_problem_removed": 0,
    }

    for index, row in enumerate(rows):
        problem = prompt_from_row(row)
        answer = answer_from_row(row)
        if not problem:
            stats["empty_problem_removed"] += 1
            continue
        if not answer:
            stats["empty_answer_removed"] += 1
            continue
        if not problem.isascii():
            stats["non_ascii_problem_removed"] += 1
            continue

        output.append(
            {
                "id": source_id(row, index),
                "student_input": problem,
                "ground_truth": answer,
                "verifier_type": "exact_match",
                "reference_solution": f"Verified final answer: \\boxed{{{answer}}}",
                "source_dataset": "DAPO-Math-14k",
            }
        )

    if max_samples > 0:
        output = output[:max_samples]
    stats["output_rows"] = len(output)
    stats["selection"] = "Official DAPO rows with nonempty answer/problem and ASCII-only problem text; source order preserved"
    return output, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = (args.source_file or output_dir / "source" / "dapo-math-17k.parquet").resolve()

    if args.source_file is None:
        download(args.source_url, source_file, args.force_download)
    elif not source_file.exists():
        raise FileNotFoundError(source_file)

    rows = load_rows(source_file)
    converted, stats = convert(rows, args.max_samples)
    if not converted:
        raise RuntimeError("Conversion produced no rows.")
    if args.max_samples <= 0 and len(converted) != EXPECTED_EOPD_ROWS:
        raise RuntimeError(
            "Official DAPO English-subset count changed: "
            f"expected {EXPECTED_EOPD_ROWS}, got {len(converted)}. "
            "Refusing to silently create a dataset different from EOPD's DAPO-Math-14k."
        )

    train_file = output_dir / "train.jsonl"
    write_jsonl(train_file, converted)
    report = {
        **stats,
        "source_url": args.source_url,
        "source_file": str(source_file),
        "source_sha256": sha256(source_file),
        "train_file": str(train_file),
        "train_sha256": sha256(train_file),
        "privileged_information": "verified final answer only; no synthetic reasoning",
    }
    with (output_dir / "conversion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
