"""
Recovers the training curves of e01b from the training logs, which e01b
did not save separately (run tracking was added afterwards, see
anchor.tracking). Writes results/{run}.curve.jsonl in the format of
RunLogger and checks every curve against the run summary: 80 points ending
at the last step, and the best validation value of each path equal to the
one in the summary.
"""

import json
import re
from pathlib import Path

from run import RESULTS_DIR

LOGS = [
    Path(__file__).resolve().parents[2] / "logs" / f"e01b_{f}.log"
    for f in ("onestep", "direct")
]
LINE = re.compile(
    r"^(?P<name>\S+) \| Step: (?P<step>\d+) \| Loss: (?P<loss>\S+) \| "
    r"Val direct: (?P<val_direct>\S+) \| Val rollout: (?P<val_rollout>\S+)$"
)
LOG_EVERY = 500
# values in the log are printed with 4 significant digits
REL_TOLERANCE = 1e-3


def parse() -> dict[str, list[dict]]:
    """Curves per run; a run that appears twice keeps its last complete pass."""
    curves = {}
    for log in LOGS:
        for line in log.read_text().splitlines():
            match = LINE.match(line)
            if match is None:
                continue
            point = {
                "step": int(match["step"]),
                **{k: float(match[k]) for k in ("loss", "val_direct", "val_rollout")},
            }
            curve = curves.setdefault(match["name"], [])
            if curve and point["step"] <= curve[-1]["step"]:
                curve.clear()
            curve.append(point)

    return curves


def main():
    curves = parse()
    problems = []
    for name, curve in sorted(curves.items()):
        with open(RESULTS_DIR / f"{name}.json") as f:
            summary = json.load(f)
        steps = [p["step"] for p in curve]
        if steps != list(range(LOG_EVERY, summary["steps"] + 1, LOG_EVERY)):
            problems.append(
                f"{name}: steps {steps[0]}..{steps[-1]}, {len(steps)} points"
            )
        for path in ("direct", "rollout"):
            best = min(p[f"val_{path}"] for p in curve)
            if (
                abs(best - summary[f"val_{path}"])
                > REL_TOLERANCE * summary[f"val_{path}"]
            ):
                problems.append(
                    f"{name}: best val_{path} {best} vs {summary[f'val_{path}']}"
                )
        with open(RESULTS_DIR / f"{name}.curve.jsonl", "w") as f:
            f.writelines(json.dumps(point) + "\n" for point in curve)

    summaries = {
        p.stem
        for p in RESULTS_DIR.glob("*.json")
        if p.stem != "test_results" and not p.stem.startswith("explore")
    }
    print(
        f"{len(curves)} curves written, {len(summaries - set(curves))} runs without a curve"
    )
    print("\n".join(problems) if problems else "every curve matches its summary")


if __name__ == "__main__":
    main()
