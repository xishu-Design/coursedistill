"""Evaluate OlympiadBench with its official deterministic judge.

The official OpenBMB ``AutoScoringJudge`` is required instead of silently
switching to a different grader. Point ``--olympiadbench_eval_dir`` or the
``OLYMPIADBENCH_EVAL_DIR`` environment variable to the public benchmark's
``eval`` directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from eval_benchmark import load_official_olympiadbench_judge, main


def _option_value(argv: list[str], option: str) -> str | None:
    for index, argument in enumerate(argv):
        if argument == option and index + 1 < len(argv):
            return argv[index + 1]
        prefix = f"{option}="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def _contains_official_judge(eval_dir: str | None) -> bool:
    if not eval_dir:
        return False
    candidate = Path(eval_dir)
    return (
        (candidate / "auto_scoring_judge.py").is_file()
        or (candidate / "eval" / "auto_scoring_judge.py").is_file()
    )


def _fixed_olympiadbench_argv(argv: list[str]) -> list[str]:
    if any(argument.split("=", 1)[0] == "--benchmark" for argument in argv):
        raise SystemExit(
            "--benchmark is not accepted by eval_olympiadbench.py; "
            "this entry point fixes the OlympiadBench protocol."
        )

    if not any(argument in {"-h", "--help"} for argument in argv):
        eval_dir = _option_value(argv, "--olympiadbench_eval_dir")
        eval_dir = eval_dir or os.environ.get("OLYMPIADBENCH_EVAL_DIR")
        if eval_dir is None:
            for candidate in (
                Path("third_party/OlympiadBench/eval"),
                Path("third_party/OlympiadBench"),
            ):
                if candidate.exists():
                    eval_dir = str(candidate)
                    break

        if not _contains_official_judge(eval_dir):
            raise SystemExit(
                "Official OlympiadBench AutoScoringJudge was not found. Set "
                "--olympiadbench_eval_dir or OLYMPIADBENCH_EVAL_DIR to the "
                "public OlympiadBench eval directory."
            )
        if load_official_olympiadbench_judge(eval_dir) is None:
            raise SystemExit(
                "The official OlympiadBench judge exists but could not be "
                "imported. Install its public evaluation dependencies."
            )

    fixed = [*argv, "--benchmark", "olympiadbench"]
    if "--use_official_olympiadbench_judge" not in fixed:
        fixed.append("--use_official_olympiadbench_judge")
    return fixed


if __name__ == "__main__":
    sys.argv[1:] = _fixed_olympiadbench_argv(sys.argv[1:])
    main()
