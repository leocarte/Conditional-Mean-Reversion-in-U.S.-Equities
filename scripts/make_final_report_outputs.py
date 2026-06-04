"""Generate final report-ready walk-forward tables and figures.

This is a post-processing script only. It consumes accepted canonical files
under ``data/outputs/`` and writes report-facing CSV/PNG artefacts under
``report/tables/`` and ``report/figures/`` when those inputs are present.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlfinance.eval.final_report_outputs import (  # noqa: E402
    build_final_report_paths,
    generate_final_report_outputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root containing data/outputs/ and report/.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = build_final_report_paths(repo_root=args.repo_root)
    result = generate_final_report_outputs(paths)

    if result.written:
        print("Wrote:")
        for path in result.written:
            print(f"  - {path.relative_to(paths.repo_root)}")
    else:
        print("No report outputs were written.")

    if result.skipped:
        print("Skipped:")
        for message in result.skipped:
            print(f"  - {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
