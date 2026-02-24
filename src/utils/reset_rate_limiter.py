"""Utility to reset the rate limiter persisted state.

Can be used as a module or run as a script to remove the
`.rate_limit.json` file used by `RateLimiter` (created in the
pipeline output directory). Removing the file causes the
rate limiter to start fresh next time the pipeline is run.

Usage (CLI):
  python -m src.utils.reset_rate_limiter --help

Functions:
  reset_rate_limiter(state_path=None, backup=True) -> Path | None
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def reset_rate_limiter(state_path: Optional[Path] = None, backup: bool = True) -> Optional[Path]:
    """Reset the persisted rate limiter state.

    Args:
        state_path: Path to the state file. If None, uses
            `<output_dir>/.rate_limit.json` from settings.
        backup: If True and the file exists, create a timestamped
            backup next to the file before removing it.

    Returns:
        The path of the backed-up file if backup was performed,
        otherwise None.
    """
    settings = get_settings()

    if state_path is None:
        out_dir = Path(settings.paths.output_dir)
        state_path = out_dir / ".rate_limit.json"

    state_path = Path(state_path)

    if not state_path.exists():
        logger.info("No rate limiter state file found at %s", state_path)
        return None

    backup_path = None
    try:
        if backup:
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = state_path.with_name(f".rate_limit_bak_{ts}.json")
            shutil.copy2(state_path, backup_path)
            logger.info("Backed up %s -> %s", state_path, backup_path)

        # Remove the state file so RateLimiter will start fresh
        state_path.unlink()
        logger.info("Removed rate limiter state file: %s", state_path)
        return backup_path

    except Exception as e:
        logger.error("Failed to reset rate limiter state: %s", e)
        raise


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reset_rate_limiter",
        description="Reset the persisted rate limiter state file",
    )

    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to the rate limiter state file (overrides output dir)",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup before removing the file",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)

    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    path = Path(args.state_file) if args.state_file else None
    try:
        backup_path = reset_rate_limiter(state_path=path, backup=not args.no_backup)
        if backup_path and not args.quiet:
            print(f"Backed up state file to: {backup_path}")
        elif not args.quiet:
            print("Rate limiter state removed (no previous state found or removed).")
        return 0
    except Exception as e:
        if not args.quiet:
            print(f"Failed to reset rate limiter: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
