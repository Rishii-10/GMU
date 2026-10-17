"""Shared output-location resolver for the classifier eval scripts.

Historically evaluate_1000.py / evaluate_balanced.py / evaluate_100_cases.py
wrote their result + log JSONs straight into tests/, silently overwriting the
git-tracked baseline files on every run. That made "did the numbers change?"
impossible to answer from git and let an accidental run clobber a committed
baseline.

Default behaviour now: write into a fresh timestamped directory under
tests/eval_runs/ (gitignored), so a run never touches the tracked baselines.
Opt in to updating the baselines with --write, or choose a location with
--out-dir DIR.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def _build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--write",
        action="store_true",
        help="overwrite the tracked baseline JSONs in tests/ (git-tracked)",
    )
    group.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="write results into this directory instead of the default timestamped run dir",
    )
    # Used by the harnesses that call a live LLM once per case
    # (evaluate_llm.py, evaluate_end_to_end.py), where a full pass is hundreds
    # of model calls and a capped pass is wanted for a smoke test. Declared
    # here rather than sniffed out of sys.argv by each script, so --help
    # documents it and an unknown flag is still a hard error.
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score at most this many cases (LLM-backed harnesses only)",
    )
    return parser


def _out_dir_from(args: argparse.Namespace) -> Path:
    if args.out_dir:
        out = Path(args.out_dir)
    elif args.write:
        out = TESTS_DIR
    else:
        out = TESTS_DIR / "eval_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_out_dir(description: str) -> Path:
    """Parse CLI args and return the directory the eval should write into.

    --write       -> tests/ (overwrite the git-tracked baseline JSONs)
    --out-dir DIR -> DIR
    (neither)     -> tests/eval_runs/<timestamp>/  (default, never tracked)
    """
    return _out_dir_from(_build_parser(description).parse_args())


def resolve_out_dir_and_limit(description: str) -> tuple[Path, "int | None"]:
    """`resolve_out_dir` plus the parsed `--limit` (None when not given).

    Separate entry point so the existing callers keep their simple return
    type; LLM-backed harnesses use this one.
    """
    args = _build_parser(description).parse_args()
    return _out_dir_from(args), args.limit
