"""Disposable launcher diagnostics; no scientific jobs or model execution."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "Libs/scripts/run_rumordiffusion_ablation.sh"
VARIANTS = ("full", "without_reg", "without_pair", "without_temporal", "without_text", "without_temperature")
FILES = (
    "rumordiffusion_full.yaml", "rumordiffusion_without_regularization.yaml",
    "rumordiffusion_without_pair.yaml", "rumordiffusion_without_temporal.yaml",
    "rumordiffusion_without_text.yaml", "rumordiffusion_without_temperature.yaml",
)
SEEDS = (41, 42, 43)


@pytest.fixture
def launcher(tmp_path):
    runtime = tmp_path / "runtime"
    source = tmp_path / "source"
    binary = tmp_path / "bin"
    binary.mkdir()
    (runtime / "Libs/configs").mkdir(parents=True)
    models = source / "Libs/configs/model_canonical"
    models.mkdir(parents=True)
    checkpoint_root = runtime / "Output/exploratory/rumordiffusion_canonical_v1/ckpts/manuscript_main"
    (runtime / "Libs/configs/exp_manuscript_three_seed.yaml").write_text(
        json.dumps({"outputs": {"ckpt_dir": str(checkpoint_root)}}), encoding="utf-8",
    )
    for variant, filename in zip(VARIANTS, FILES):
        (models / filename).write_text(json.dumps({"name": "rumordiffusion", "variant": variant}), encoding="utf-8")
    (models / "rumordiffusion.yaml").write_text(
        '{"name":"rumordiffusion","variant":"canonical"}', encoding="utf-8",
    )
    log = tmp_path / "calls.jsonl"
    git = binary / "git"
    git.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ["FIXTURE_CALL_LOG"], "a") as stream:
    stream.write(json.dumps({"tool":"git", "args":sys.argv[1:]})+"\\n")
if "rev-parse" in sys.argv:
    print(os.environ["FIXTURE_HEAD"])
elif "status" in sys.argv:
    print(os.environ.get("FIXTURE_DIRTY", ""), end="")
""", encoding="utf-8")
    python = binary / "fixture-python"
    python.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
program = sys.stdin.read() if args and args[0] == "-" else ""
with open(os.environ["FIXTURE_CALL_LOG"], "a") as stream:
    stream.write(json.dumps({"tool":"python", "args":args, "program":program, "cwd":os.getcwd(),
                             "tmpdir":os.environ.get("TMPDIR"),
                             "pythonpath":os.environ.get("PYTHONPATH"),
                             "output_root":os.environ.get("DPLUS_OUTPUT_ROOT"),
                             "data_root":os.environ.get("RUMORDIFFUSION_PROCESSED_ROOT")})+"\\n")
if args and args[0] == "-":
    experiment = json.loads(pathlib.Path(args[1]).read_text())
    model = json.loads(pathlib.Path(args[2]).read_text())
    variant = model.get("variant", model["name"])
    subdirectory = model["name"] if variant == "canonical" else os.path.join(model["name"], variant)
    print(os.path.join(experiment["outputs"]["ckpt_dir"], args[3], subdirectory, "seed"+args[4], "best.pt"))
elif "Libs.exp.manuscript_main_evidence" in args:
    sys.exit(int(os.environ.get("FIXTURE_GATE_EXIT", "0")))
""", encoding="utf-8")
    git.chmod(0o755)
    python.chmod(0o755)
    environment = dict(os.environ)
    for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        environment.pop(key, None)
    environment.update(
        PATH=f"{binary}:{environment['PATH']}", PYTHON_BIN=str(python),
        DPLUS_REPO_ROOT=str(runtime), DPLUS_MANUSCRIPT_SOURCE_ROOT=str(source),
        DPLUS_MANUSCRIPT_CODE_COMMIT="a" * 40,
        DPLUS_MANUSCRIPT_QUALIFICATION_SOURCE_ROOT=str(source), DPLUS_MANUSCRIPT_QUALIFICATION_CODE_COMMIT="a" * 40,
        DPLUS_MANUSCRIPT_EXECUTION_PROVENANCE=str(runtime / "Libs/configs/exp_manuscript_three_seed.yaml"), DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT="b" * 40,
        FIXTURE_HEAD="a" * 40, FIXTURE_CALL_LOG=str(log), SLURM_CPUS_PER_TASK="1",
    )

    def run(index=0, *, dry=True, script=SCRIPT, slurm=None, changes=None):
        env = dict(environment)
        env["SLURM_ARRAY_TASK_ID"] = str(index)
        if slurm if slurm is not None else not dry:
            env["SLURM_JOB_ID"] = "424242"
        if changes:
            for key, value in changes.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = str(value)
        if log.exists():
            log.unlink()
        result = subprocess.run(
            ["bash", str(script), *(["--dry-run"] if dry else [])],
            cwd=runtime, env=env, text=True, capture_output=True,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run, runtime, source


def _modules(calls, name):
    return [call for call in calls if call["tool"] == "python" and name in call["args"]]


def test_all_sixty_task_identities_and_three_seeds_are_preserved(launcher):
    run, _, _ = launcher
    identities = set()
    for index in range(36):
        result, _ = run(index)
        assert result.returncode == 0, result.stderr
        expected = (("twitter15", "twitter16")[index // 18], VARIANTS[(index % 18) // 3], SEEDS[index % 3])
        assert result.stdout.splitlines()[0] == (
            f"dataset={expected[0]} variant={expected[1]} seed={expected[2]} task={index} total=36"
        )
        identities.add(expected)
    assert len(identities) == 36
    assert {seed for _, _, seed in identities} == set(SEEDS)
    text = SCRIPT.read_text()
    assert "#SBATCH --array=0-35%1" in text
    assert "#SBATCH --nodelist=cpu04" in text
    assert "#SBATCH --cpus-per-task=8" in text and "#SBATCH --mem=48G" in text


@pytest.mark.parametrize("index", [0, 15, 18, 33])
def test_full_and_temperature_use_the_qualified_main_checkpoint_without_training(launcher, index):
    run, runtime, source = launcher
    result, calls = run(index, dry=False)
    assert result.returncode == 0, result.stderr
    gate = _modules(calls, "Libs.exp.manuscript_main_evidence")
    evaluation = _modules(calls, "Libs.exp.run")
    assert len(gate) == len(evaluation) == 1
    assert gate[0]["args"][-4:] == ["--case", ("twitter15", "twitter16")[index // 18], "rumordiffusion", "41"]
    assert gate[0]["args"][gate[0]["args"].index("--training-code-commit") + 1] == "b" * 40
    args = evaluation[0]["args"]
    for flag in ("--eval_only", "--canonical_early_only", "--canonical_artifacts_only"):
        assert flag in args
    assert "--canonical_main_only" not in args
    checkpoint = Path(args[args.index("--ckpt") + 1])
    assert checkpoint.parent.parent.name == "rumordiffusion"
    assert evaluation[0]["cwd"] == str(source)
    assert evaluation[0]["output_root"] == str(runtime / "Output")
    assert evaluation[0]["data_root"] == str(runtime / "Input/processed_canonical_v1")
    assert not Path(evaluation[0]["tmpdir"]).exists()


@pytest.mark.parametrize("index", [3, 6, 9, 12])
def test_corrected_regularization_and_other_training_variants_run_main_then_early(launcher, index):
    run, _, _ = launcher
    result, calls = run(index, dry=False)
    assert result.returncode == 0, result.stderr
    gate = _modules(calls, "Libs.exp.manuscript_main_evidence")
    assert len(gate) == (1 if index == 6 else 0)
    commands = _modules(calls, "Libs.exp.run")
    assert len(commands) == 2
    assert "--canonical_main_only" in commands[0]["args"] and "--eval_only" not in commands[0]["args"]
    for flag in ("--eval_only", "--canonical_early_only", "--canonical_artifacts_only"):
        assert flag in commands[1]["args"]
    assert not Path(commands[1]["tmpdir"]).exists()


@pytest.mark.parametrize("index", [0, 6, 15])
def test_full_gate_failure_stops_before_model_execution_and_cleans_scratch(launcher, index):
    run, _, _ = launcher
    result, calls = run(index, dry=False, changes={"FIXTURE_GATE_EXIT": "7"})
    assert result.returncode == 7
    assert not _modules(calls, "Libs.exp.run")
    assert not Path(_modules(calls, "Libs.exp.manuscript_main_evidence")[0]["tmpdir"]).exists()


@pytest.mark.parametrize("changes", [
    {"DPLUS_MANUSCRIPT_CODE_COMMIT": None},
    {"DPLUS_MANUSCRIPT_CODE_COMMIT": "c" * 40},
    {"FIXTURE_DIRTY": " M Libs/exp/run.py\n"},
])
def test_unapproved_or_dirty_source_is_rejected_before_python_execution(launcher, changes):
    run, _, _ = launcher
    result, calls = run(3, dry=False, changes=changes)
    assert result.returncode == 2
    assert "clean" in result.stderr and ("source commit" in result.stderr or "qualification source" in result.stderr)
    assert not [call for call in calls if call["tool"] == "python"]


def test_execution_outside_slurm_is_rejected(launcher):
    run, _, _ = launcher
    result, calls = run(3, dry=False, slurm=False)
    assert result.returncode == 2 and "authorized Slurm job" in result.stderr
    assert not calls


def test_copied_spool_script_uses_runtime_root_and_configurable_source(launcher, tmp_path):
    run, runtime, source = launcher
    spool = tmp_path / "spool" / "job424242" / "slurm_script"
    spool.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, spool)
    result, calls = run(12, dry=False, script=spool, changes={"DPLUS_REPO_ROOT": None})
    assert result.returncode == 0, result.stderr
    commands = _modules(calls, "Libs.exp.run")
    assert commands[0]["args"][commands[0]["args"].index("--exp") + 1] == str(runtime / "Libs/configs/exp_manuscript_three_seed.yaml")
    assert commands[0]["pythonpath"] == str(source)


@pytest.mark.parametrize("index", [-1, 36, "invalid"])
def test_invalid_array_indices_are_rejected(launcher, index):
    run, _, _ = launcher
    result, calls = run(index)
    assert result.returncode == 2 and "Invalid manuscript ablation task index" in result.stderr
    assert not calls


@pytest.mark.parametrize("index", [0, 6, 15])
def test_full_reuse_requires_an_explicit_training_commit(launcher, index):
    run, _, _ = launcher
    result, calls = run(index, changes={"DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT": None})
    assert result.returncode == 2 and "DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT" in result.stderr
    assert not calls


def test_checkpoint_resolver_uses_the_actual_established_helper_without_reading_datasets(launcher, monkeypatch):
    run, _, source = launcher
    result, calls = run(3)
    assert result.returncode == 0
    resolver = next(call for call in calls if call["tool"] == "python" and call["args"][0] == "-")
    assert resolver["cwd"] == str(source)
    assert "resolve_model_checkpoint_subdir" in resolver["program"]
    monkeypatch.setattr(sys, "argv", resolver["args"])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(resolver["program"], "checkpoint-resolver", "exec"), {})
    # Compare the real helper result to the path printed by the disposable driver.
    assert output.getvalue().strip() in result.stdout
