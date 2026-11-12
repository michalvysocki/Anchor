import json

from anchor.tracking import RunLogger


def test_run_logger_writes_curve_and_summary(tmp_path):
    tracker = RunLogger("run", tmp_path, {"lambda": 0.1}, "cpu")
    tracker.log(500, loss=1.0, val=2.0)
    tracker.log(1000, loss=0.5, val=1.5)
    summary = tracker.finish({"val_best": 1.5})

    lines = (tmp_path / "run.curve.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [500, 1000]
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved == summary
    assert saved["lambda"] == 0.1 and saved["val_best"] == 1.5
    for key in ("git_commit", "git_dirty", "device", "torch", "started", "seconds"):
        assert key in saved["metadata"]


def test_run_logger_restarts_curve(tmp_path):
    RunLogger("run", tmp_path, {}, "cpu").log(1, loss=1.0)
    RunLogger("run", tmp_path, {}, "cpu")
    assert (tmp_path / "run.curve.jsonl").read_text() == ""
