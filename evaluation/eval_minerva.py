"""Evaluate Minerva Math with the benchmark-specific official-style protocol.

This entry point fixes the benchmark mode to Minerva Math. Generation is
shared with ``eval_benchmark.py``, while answer extraction and grading use its
Minerva implementation: the last boxed answer, lm-evaluation-harness-style
normalization, symbolic equivalence, and the benchmark's numeric tolerance.
"""

from __future__ import annotations

import sys

from eval_benchmark import main


def _fixed_minerva_argv(argv: list[str]) -> list[str]:
    forbidden = {"--benchmark", "--use_official_olympiadbench_judge"}
    for argument in argv:
        name = argument.split("=", 1)[0]
        if name in forbidden or name == "--olympiadbench_eval_dir":
            raise SystemExit(
                f"{name} is not accepted by eval_minerva.py; "
                "this entry point fixes the Minerva Math protocol."
            )
    return [*argv, "--benchmark", "minerval"]


if __name__ == "__main__":
    sys.argv[1:] = _fixed_minerva_argv(sys.argv[1:])
    main()
