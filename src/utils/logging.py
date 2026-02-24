import logging
import re
import sys
from pathlib import Path


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text.

    Removes sequences like [31m, [1;33m, etc.
    """
    ansi_escape = re.compile(r"\033\[[0-9;]*m")
    return ansi_escape.sub("", text)


class ColoredFormatter(logging.Formatter):
    """Custom logging formatter that adds colors and icons based on level and logger name."""

    # ANSI Escape Codes
    RESET = "\033[0m"
    BOLD = "\033[1m"

    COLORS = {
        "DEBUG": "\033[37m",  # White
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }

    # Icons and Colors for specific components
    AGENT_THEMES = {
        # Core Agents
        "src.agents.orchestrator": ("🤖", "\033[1;38;5;202m"),  # Bold Orange
        "src.agents.workers.geo_analyzer": ("📍", "\033[1;32m"),  # Bold Green
        "src.agents.workers.temporal_analyzer": ("⏳", "\033[1;33m"),  # Bold Yellow
        "src.agents.workers.type_analyzer": ("🏛️ ", "\033[1;35m"),  # Bold Magenta
        "src.agents.workers.path_generator": ("🗺️ ", "\033[1;36m"),  # Bold Cyan
        # Pipeline Stages
        "src.pipeline": ("⚙️ ", "\033[1;34m"),  # Bold Blue
        "src.loaders": ("📂", "\033[1;36m"),  # Bold Cyan (Loaders)
        "src.workflow": ("🔄", "\033[1;35m"),  # Bold Magenta (Workflow/Graph)
        "src.triples.generator": ("🔗", "\033[1;32m"),  # Bold Green
        "src.triples.validator": ("✅", "\033[1;33m"),  # Bold Yellow
        # Infrastructure
        "src.llm": ("🧠", "\033[1;38;5;220m"),  # Bold Gold (LLM)
        "src.utils": ("🛠️ ", "\033[1;90m"),  # Dark Gray (Utils)
        "src.agents.base": ("⚙️ ", "\033[1;90m"),  # Dark Gray (Agent Base)
        "src.main": ("🚀", "\033[1;32m"),  # Bold Green (Main)
        "__main__": ("🚀", "\033[1;32m"),  # Bold Green (Main)
        "root": ("⚙️ ", "\033[1;90m"),  # Dark Gray (Root)
    }

    def format(self, record):
        # Get theme for the logger
        icon, agent_color = "", ""
        for name, theme in self.AGENT_THEMES.items():
            # Match both 'src.module' and 'module' (resilient to different import styles)
            alt_name = name.replace("src.", "") if name.startswith("src.") else name
            if record.name.startswith(name) or record.name.startswith(alt_name):
                icon, agent_color = theme
                break

        # Fallback for other loggers
        if not icon:
            icon = "•"
            agent_color = self.BOLD

        # Format level
        level_color = self.COLORS.get(record.levelname, self.RESET)
        level_name = f"{level_color}{record.levelname:8}{self.RESET}"

        # Format name (agent)
        short_name = record.name.split(".")[-1]
        agent_display = f"{agent_color}{icon} {short_name:18}{self.RESET}"

        # Format message
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            message = f"{level_color}{message}{self.RESET}"

        # Combine
        timestamp = self.formatTime(record, self.datefmt)
        return f"{timestamp} | {level_name} | {agent_display} | {message}"


class PlainFormatter(logging.Formatter):
    """Plain text formatter for file logging (no ANSI codes)."""

    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)
        short_name = record.name.split(".")[-1]
        message = strip_ansi_codes(record.getMessage())
        return f"{timestamp} | {record.levelname:8} | {short_name:18} | {message}"


# Global file handler reference (to allow adding it later)
_file_handler: logging.FileHandler | None = None


def setup_colored_logging(level=logging.INFO, log_file: str | Path | None = None):
    """
    Sets up global logging with the ColoredFormatter.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file for persistent logging
    """
    global _file_handler

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)

    # File handler (plain text, no colors)
    if log_file:
        add_file_handler(log_file, level)

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def add_file_handler(log_file: str | Path, level=logging.DEBUG) -> logging.FileHandler:
    """
    Add a file handler to the root logger.

    Args:
        log_file: Path to the log file
        level: Logging level for file (default: DEBUG for maximum detail)

    Returns:
        The created FileHandler
    """
    global _file_handler

    # Remove existing file handler if present
    if _file_handler:
        remove_file_handler()

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    _file_handler.setLevel(level)
    _file_handler.setFormatter(PlainFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    logging.getLogger().addHandler(_file_handler)
    logging.info(f"File logging enabled: {log_path}")

    return _file_handler


def remove_file_handler() -> None:
    """Remove the file handler from the root logger."""
    global _file_handler

    if _file_handler:
        logging.getLogger().removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None


def get_file_handler() -> logging.FileHandler | None:
    """Get the current file handler (if any)."""
    return _file_handler
