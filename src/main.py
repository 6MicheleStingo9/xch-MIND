#!/usr/bin/env python3
"""
xch-MIND CLI - Command-line interface for xch-MIND pipeline.

Usage:
    python -m src.main --help
    python -m src.main --limit 5 --dry-run
    python -m src.main --format json-ld --output ./output
"""

import argparse
import logging
import sys
from pathlib import Path

import pyfiglet
from src.utils.logging import setup_colored_logging

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.config.settings import get_settings
from src.pipeline import Pipeline, PipelineResult


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging based on verbosity level."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    setup_colored_logging(level=level)


def print_banner() -> None:
    """Print the application banner."""

    isometric1_text = pyfiglet.figlet_format("xch-Mind", font="isometric1", width=100)
    print("".center(100, "*"))
    print(isometric1_text)
    print("Multi-agent Interpretive Nexus Discovery".center(100, "*"))


def print_config_summary(settings, args: argparse.Namespace, pipeline=None) -> None:
    """Print configuration summary."""
    print("\n📋 Configuration:")
    print("─" * 40)
    print(f"  LLM Provider: {settings.llm.provider}")

    if settings.llm.provider == "gemini":
        print(f"  Model: {settings.llm.gemini.model}")
    else:
        print(f"  Model: {settings.llm.ollama.model}")
        print(f"  Ollama URL: {settings.llm.ollama.base_url}")

    print(f"  Entities dir: {settings.paths.entities_dir}")
    print(f"  Output dir: {args.output or settings.paths.output_dir}")
    print(f"  Output format: {args.format or settings.output.format}")

    if args.limit:
        print(f"  Entity limit: {args.limit}")
    else:
        print("  Entity limit: None (all)")

    if args.dry_run:
        print("  Mode: 🧪 DRY RUN (no LLM calls)")
    else:
        print("  Mode: 🚀 LIVE (with LLM calls)")

    print("─" * 40)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="xch-MIND",
        description="xch-MIND - Multi-agent Interpretive Nexus Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 5 entities in dry-run mode (no LLM calls)
  python -m src.main --limit 5 --dry-run

  # Run full pipeline with Turtle output
  python -m src.main --format turtle

  # Run with custom output directory
  python -m src.main --output ./my_output --format json-ld

  # Run with Ollama instead of Gemini
  python -m src.main --provider ollama --limit 3

  # Verbose output for debugging
  python -m src.main --limit 2 --verbose --dry-run
        """,
    )

    # Entity selection
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of entities to process (default: all)",
    )

    parser.add_argument(
        "--entities-dir",
        "-e",
        type=str,
        default=None,
        help="Directory containing entity XML files (default: ./entities)",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for generated files (default: ./output)",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["turtle", "json-ld", "xml"],
        default=None,
        help="Output format (default: from config)",
    )

    # LLM options
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        choices=["gemini", "ollama"],
        default=None,
        help="LLM provider to use (default: from config)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model name to use (default: from config)",
    )

    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama server URL (default: http://localhost:11434)",
    )

    # Execution modes
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Simulate execution without LLM calls",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip RDF validation step",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (very verbose)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
    )

    # Misc
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    parser.add_argument(
        "--disable-rate-limiter",
        action="store_true",
        help="Disable LLM rate limiting (override config)",
    )

    return parser


def apply_cli_overrides(settings, args: argparse.Namespace) -> None:
    """Apply CLI arguments to settings."""
    if args.provider:
        settings.llm.provider = args.provider

    if args.model:
        if settings.llm.provider == "gemini":
            settings.llm.gemini.model = args.model
        else:
            settings.llm.ollama.model = args.model

    if args.ollama_url:
        settings.llm.ollama.base_url = args.ollama_url

    if args.entities_dir:
        settings.paths.entities_dir = Path(args.entities_dir)

    if args.output:
        settings.paths.output_dir = Path(args.output)

    if args.format:
        settings.output.format = args.format

    # Allow disabling the rate limiter from the CLI
    if getattr(args, "disable_rate_limiter", False):
        settings.llm.rate_limiting.enabled = False


def run_with_progress(pipeline: Pipeline, args: argparse.Namespace) -> PipelineResult:
    """
    Run pipeline with progress indicators.

    Uses the pipeline.execute() method which handles:
    - Run directory creation (output/xch_run_YYYYMMDD_HHMMSS/)
    - File logging (execution.log)
    - Metadata saving (metadata.json)
    - Turtle and JSON-LD output files
    """
    print("\n🚀 Starting pipeline execution...")

    if args.dry_run:
        print("   [DRY RUN] Simulating LLM calls...")

    # Use the unified execute() method
    result = pipeline.execute(
        limit=args.limit,
        dry_run=args.dry_run,
        output_format=args.format,
        skip_validation=args.skip_validation,
        enable_file_logging=True,
    )

    return result


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        setup_logging(verbose=False, debug=False)
        logging.disable(logging.CRITICAL)
    else:
        setup_logging(verbose=args.verbose, debug=args.debug)

    # Print banner unless quiet
    if not args.quiet:
        print_banner()

    # Load and configure settings
    try:
        settings = get_settings()
        apply_cli_overrides(settings, args)
    except Exception as e:
        print(f"❌ Configuration error: {e}", file=sys.stderr)
        return 1

    # Create pipeline early to get quota info
    try:
        pipeline = Pipeline(
            settings=settings,
            entities_dir=args.entities_dir,
            output_dir=args.output,
        )
    except Exception as e:
        print(f"❌ Pipeline initialization error: {e}", file=sys.stderr)
        return 1

    # Print config summary with quota info
    if not args.quiet:
        print_config_summary(settings, args, pipeline=pipeline)

    # Run pipeline
    try:

        result = run_with_progress(pipeline, args)

        # Print summary
        if not args.quiet:
            result.print_summary()

        # Return code based on result
        if result.validation_errors:
            return 2  # Validation errors

        return 0  # Success

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130

    except Exception as e:
        # Check if it's a daily quota exhaustion
        from src.llm.provider import DailyQuotaExhaustedError

        if isinstance(e, DailyQuotaExhaustedError) or (
            e.__cause__ and isinstance(e.__cause__, DailyQuotaExhaustedError)
        ):
            print("\n" + "=" * 60)
            print("❌ DAILY API QUOTA EXHAUSTED")
            print("=" * 60)
            print("The daily API limit has been reached.")
            print("Please wait until tomorrow or upgrade to a paid plan.")
            print("=" * 60 + "\n")
            return 3  # Quota exhaustion

        logging.exception("Pipeline failed")
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
