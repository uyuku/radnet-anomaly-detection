from pathlib import Path
import subprocess
import sys

import pytest

import run_pipeline as runner


def test_full_plan_has_no_unresolved_generated_dependencies(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text("random_seed: 42", encoding="utf-8")
    assert runner.missing_inputs(runner.build_plan(), root=tmp_path) == {}
    names = [step.script for step in runner.build_plan()]
    assert names.index("download_pilot_data.py") < names.index("parse_stations.py")
    assert names.index("compute_train_only_baselines.py") < names.index("synthetic_injection.py")
    assert names.index("train_and_evaluate_rigorous.py") < names.index("plot_model_evaluation.py")


def test_preflight_requires_every_station_file(tmp_path, monkeypatch):
    steps = runner.build_plan(4)
    for name in steps[0].inputs[:-1]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("input", encoding="utf-8")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kw: pytest.fail("Must preflight before executing"))
    assert runner.execute_plan(steps, root=tmp_path) == 2


def test_child_failure_stops_pipeline_and_propagates_exit_code(tmp_path, capsys):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "fail.py").write_text("raise SystemExit(7)", encoding="utf-8")
    steps = [runner.Step(0, "fail.py"), runner.Step(0, "must_not_run.py")]
    assert runner.execute_plan(steps, root=tmp_path) == 7
    assert "Pipeline complete" not in capsys.readouterr().out


def test_missing_outputs_are_a_failure(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "noop.py").write_text("pass", encoding="utf-8")
    assert runner.execute_plan([runner.Step(0, "noop.py", outputs=("missing.csv",))], root=tmp_path) == 1


def test_child_runs_with_project_working_directory(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "write.py").write_text("from pathlib import Path\nPath('result.txt').write_text('ok')", encoding="utf-8")
    assert runner.execute_plan([runner.Step(0, "write.py", outputs=("result.txt",))], root=tmp_path) == 0
    assert (tmp_path / "result.txt").read_text() == "ok"


@pytest.mark.parametrize("args,code", [(["--phase", "9"], 2), (["--phase"], 2), (["--dry-run"], 0)])
def test_runner_cli_from_another_directory(tmp_path, args, code):
    result = subprocess.run([sys.executable, str(Path(runner.__file__)), *args], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == code
    if code == 0:
        assert "compute_train_only_baselines.py" in result.stdout
        assert "Plan only" in result.stdout
        assert "Pipeline complete" not in result.stdout
