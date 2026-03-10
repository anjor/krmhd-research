"""Run management utilities: ID generation, logging, config helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def generate_run_id(study: str, param_label: str) -> str:
    """Generate a unique run ID.

    Format: {study}_{param_label}_{YYYYMMDD_HHMMSS}
    Example: 01_M004_20260310_143022
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{study}_{param_label}_{timestamp}"


def log_run(
    run_id: str,
    config_path: str,
    hardware: str,
    wall_time: float,
    outcome: str,
    notes: str = "",
    log_file: str = "docs/run_log.md",
) -> None:
    """Append a run entry to the run log.

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    config_path : str
        Path to the YAML config used.
    hardware : str
        Hardware description (e.g. "M1 Pro, 16GB").
    wall_time : float
        Wall clock time in seconds.
    outcome : str
        "pass" or "fail".
    notes : str
        Additional notes.
    log_file : str
        Path to the run log markdown file.
    """
    log_path = Path(log_file)
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "# Run Log\n\n"
            "| Run ID | Config | Hardware | Wall Time (s) | Outcome | Notes |\n"
            "|--------|--------|----------|---------------|---------|-------|\n"
        )

    line = f"| {run_id} | {config_path} | {hardware} | {wall_time:.1f} | {outcome} | {notes} |\n"
    with open(log_path, "a") as f:
        f.write(line)


def detect_hardware() -> str:
    """Return a short hardware description string."""
    import platform
    machine = platform.machine()
    processor = platform.processor() or "unknown"
    return f"{platform.system()} {machine} ({processor})"
