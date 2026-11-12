"""
Run tracking: a JSONL training curve and a summary with metadata for every
run, written next to the results. Local files are the source of truth;
Weights & Biases is an optional mirror (offline unless WANDB_MODE says
otherwise) that is only imported when requested.
"""

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

WANDB_PROJECT = "anchor"


def git_state() -> dict:
    """Commit of the working tree and whether it has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}

    return {"git_commit": commit, "git_dirty": dirty}


def environment(device: str) -> dict:
    """Machine and library versions of the run."""
    return {
        "device": device,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


class RunLogger:
    """
    Logs one training run.

    Writes `{name}.curve.jsonl` (one line per log call) and, on finish,
    `{name}.json` with the config, the final values and the metadata.

    Attributes:
        name (str): run name, also the file stem
        results_dir (Path): directory of the run files
        config (dict): settings of the run, copied into the summary
    """

    def __init__(
        self,
        name: str,
        results_dir: Path,
        config: dict,
        device: str,
        use_wandb: bool = False,
        group: str | None = None,
    ):
        self.name = name
        self.results_dir = Path(results_dir)
        self.config = config
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.curve_path = self.results_dir / f"{name}.curve.jsonl"
        self.curve_path.write_text("")
        self.metadata = {
            **git_state(),
            **environment(device),
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self._start = time.monotonic()
        self._wandb = None
        if use_wandb:
            import wandb

            os.environ.setdefault("WANDB_MODE", "offline")
            self._wandb = wandb.init(
                project=WANDB_PROJECT,
                group=group,
                name=name,
                config={**config, **self.metadata},
                # offline runs go next to the experiment's results, not the cwd
                dir=self.results_dir.parent,
                reinit=True,
            )

    def log(self, step: int, **metrics: float):
        """Appends one point of the training curve."""
        with open(self.curve_path, "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self, results: dict) -> dict:
        """Writes the summary and returns it."""
        self.metadata["seconds"] = round(time.monotonic() - self._start, 1)
        summary = {**self.config, **results, "metadata": self.metadata}
        with open(self.results_dir / f"{self.name}.json", "w") as f:
            json.dump(summary, f)
        if self._wandb is not None:
            self._wandb.summary.update(results)
            self._wandb.finish()

        return summary
