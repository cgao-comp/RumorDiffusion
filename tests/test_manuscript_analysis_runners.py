"""Disposable Slurm-launcher checks; no scientific jobs or model inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "Libs/scripts"


@pytest.fixture(params=("sensitivity", "threshold"))
def launcher(tmp_path, request):
    analysis = request.param
    script_path = SCRIPT_ROOT / f"run_manuscript_{analysis}.sh"
    runtime, source, binary, qualification = (tmp_path / name for name in ("runtime", "source", "bin", "qualification"))
    qualification.mkdir()
    binary.mkdir()
    (runtime / "Libs/configs").mkdir(parents=True)
    (runtime / "Libs/configs/exp_manuscript_main.yaml").write_text("{}\n")
    (runtime / "Libs/configs/exp_manuscript_three_seed.yaml").write_text("{}\n")
    provenance = runtime / "seed42_execution.json"
    provenance.write_text("{}\n")
    (source / "Libs/configs/model_canonical").mkdir(parents=True)
    (source / "Libs/configs/model_canonical/rumordiffusion.yaml").write_text("{}\n")
    (source / "Libs/exp").mkdir(parents=True)
    (source / f"Libs/exp/manuscript_{analysis}_evidence.py").write_text("# disposable fixture\n")
    checkpoint = tmp_path / "main checkpoint.pt"
    checkpoint.write_bytes(b"disposable fixture")
    log = tmp_path / "calls.jsonl"
    git = binary / "git"
    git.write_text("""#!/usr/bin/env python3
import os,sys
if "rev-parse" in sys.argv:
    print(os.environ["FIXTURE_HEAD"])
elif "status" in sys.argv:
    print(os.environ.get("FIXTURE_DIRTY", ""), end="")
""")
    python = binary / "fixture-python"
    python.write_text("""#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]
mode="checkpoint" if "--source-checkpoint" in args else "case" if "--case" in args else "inference"
with open(os.environ["FIXTURE_LOG"], "a") as stream:
    stream.write(json.dumps({"mode":mode,"args":args,"cwd":os.getcwd(),"scratch":os.environ["TMPDIR"],
        "pythonpath":os.environ["PYTHONPATH"],"output":os.environ["DPLUS_OUTPUT_ROOT"],
        "cuda":os.environ["CUDA_VISIBLE_DEVICES"],"threads":os.environ["OMP_NUM_THREADS"],
        "training":os.environ["DPLUS_MANUSCRIPT_RD_TRAINING_CODE_COMMIT"],
        "calibration":os.environ["DPLUS_MANUSCRIPT_EARLY_CALIBRATION_CODE_COMMIT"],
        "inference":os.environ[os.environ["FIXTURE_INFERENCE_ENV"]]})+"\\n")
if mode==os.environ.get("FIXTURE_FAIL_MODE"):
    sys.exit(int(os.environ["FIXTURE_FAIL_CODE"]))
if mode=="checkpoint":
    print(os.environ["FIXTURE_CHECKPOINT"])
""")
    git.chmod(0o755)
    python.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DPLUS_MANUSCRIPT_", "SLURM_"))}
    env.update(PATH=f"{binary}:{env['PATH']}", PYTHON_BIN=str(python), DPLUS_REPO_ROOT=str(runtime),
               DPLUS_MANUSCRIPT_SOURCE_ROOT=str(source), DPLUS_MANUSCRIPT_CODE_COMMIT="a" * 40,
               DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT="b" * 40,
               DPLUS_MANUSCRIPT_QUALIFICATION_SOURCE_ROOT=str(qualification),
               DPLUS_MANUSCRIPT_QUALIFICATION_CODE_COMMIT="a" * 40,
               DPLUS_MANUSCRIPT_EXECUTION_PROVENANCE=str(provenance),
               DPLUS_MANUSCRIPT_EARLY_CALIBRATION_CODE_COMMIT="c" * 40,
               FIXTURE_INFERENCE_ENV=f"DPLUS_MANUSCRIPT_{analysis.upper()}_CODE_COMMIT",
               FIXTURE_HEAD="a" * 40, FIXTURE_LOG=str(log), FIXTURE_CHECKPOINT=str(checkpoint), SLURM_CPUS_PER_TASK="1")

    def run(index=0, *, dry=False, slurm=True, script=script_path, changes=None, arguments=()):
        active = dict(env, SLURM_ARRAY_TASK_ID=str(index))
        if slurm:
            active["SLURM_JOB_ID"] = "424242"
        for key, value in (changes or {}).items():
            if value is None:
                active.pop(key, None)
            else:
                active[key] = str(value)
        if log.exists():
            log.unlink()
        result = subprocess.run(["bash", str(script)] + (["--dry-run"] if dry else []) + list(arguments),
                                env=active, cwd=runtime, text=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        assert all(not Path(call["scratch"]).exists() for call in calls)
        return result, calls

    run.analysis = analysis
    run.script = script_path
    return run, runtime, source, checkpoint, tmp_path


@pytest.mark.parametrize("index", range(12))
def test_all_three_seed_four_dataset_tasks_use_qualified_checkpoint_and_output_gate(launcher, index):
    run, runtime, source, checkpoint, _ = launcher
    result, calls = run(index)
    assert result.returncode == 0, result.stderr
    dataset = ("pheme", "weibo", "twitter15", "twitter16")[index // 3]
    seed = (41, 42, 43)[index % 3]
    assert [call["mode"] for call in calls] == ["checkpoint", "inference", "case"]
    for call, flag in ((calls[0], "--source-checkpoint"), (calls[2], "--case")):
        args = call["args"]
        assert args[args.index(flag) + 1:args.index(flag) + 3] == [dataset, str(seed)]
        assert args[args.index("--training-code-commit") + 1] == "b" * 40
        assert args[args.index("--early-calibration-code-commit") + 1] == "c" * 40
        assert args[args.index(f"--{run.analysis}-inference-code-commit") + 1] == "a" * 40
    args = calls[1]["args"]
    assert {"--eval_only", "--canonical_artifacts_only", f"--manuscript_{run.analysis}_only"} <= set(args)
    assert args[args.index("--ckpt") + 1] == str(checkpoint)
    assert args[args.index("--dataset_override") + 1] == dataset
    assert args[args.index("--seeds") + 1] == str(seed)
    assert args[args.index("--exp") + 1] == str(runtime / "Libs/configs" / ("exp_manuscript_main.yaml" if seed == 42 else "exp_manuscript_three_seed.yaml"))
    assert calls[1]["cwd"] == str(source)
    assert calls[0]["cwd"] == calls[2]["cwd"] == str(runtime.parent / "qualification")
    for call in (calls[0], calls[2]):
        assert call["args"][call["args"].index("--config") + 1] == str(runtime / "Libs/configs/exp_manuscript_three_seed.yaml")
    assert all(call["pythonpath"] == str(source) for call in calls)
    assert all(call["cuda"] == "" and call["threads"] == "1" for call in calls)
    assert all(call["output"] == str(runtime / "Output") for call in calls)


@pytest.mark.parametrize("mode,code,modes", [
    ("checkpoint", 17, ["checkpoint"]), ("inference", 18, ["checkpoint", "inference"]),
    ("case", 19, ["checkpoint", "inference", "case"]),
])
def test_failed_prerequisite_or_output_gate_propagates_exit_and_cleans_scratch(launcher, mode, code, modes):
    run, *_ = launcher
    result, calls = run(changes={"FIXTURE_FAIL_MODE": mode, "FIXTURE_FAIL_CODE": code})
    assert result.returncode == code
    assert [call["mode"] for call in calls] == modes


@pytest.mark.parametrize("changes", [
    {"DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT": None}, {"DPLUS_MANUSCRIPT_EARLY_CALIBRATION_CODE_COMMIT": None},
    {"DPLUS_MANUSCRIPT_CODE_COMMIT": None}, {"FIXTURE_HEAD": "d" * 40}, {"FIXTURE_DIRTY": " M model.py\n"},
    {"DPLUS_MANUSCRIPT_EARLY_CALIBRATION_CODE_COMMIT": "short"},
])
def test_incomplete_or_conflicting_commits_prevent_execution(launcher, changes):
    run, *_ = launcher
    result, calls = run(changes=changes)
    assert result.returncode == 2 and calls == []


def test_explicit_analysis_and_training_pins_override_shared_fallbacks(launcher):
    run, *_ = launcher
    result, calls = run(changes={f"DPLUS_MANUSCRIPT_{run.analysis.upper()}_CODE_COMMIT": "d" * 40,
                                "DPLUS_MANUSCRIPT_RD_TRAINING_CODE_COMMIT": "e" * 40,
                                "FIXTURE_HEAD": "d" * 40,
                                "DPLUS_MANUSCRIPT_QUALIFICATION_CODE_COMMIT": "d" * 40})
    assert result.returncode == 0, result.stderr
    assert all(call["training"] == "e" * 40 and call["inference"] == "d" * 40 and call["calibration"] == "c" * 40 for call in calls)


@pytest.mark.parametrize("index", ["-1", "12", "01", "word"])
def test_invalid_task_index_is_rejected(launcher, index):
    run, *_ = launcher
    result, calls = run(index)
    assert result.returncode == 2 and calls == []


def test_spooled_dry_run_uses_submission_directory_without_execution(launcher):
    run, runtime, _, _, scratch = launcher
    spool = scratch / "spool" / "slurm_script"
    spool.parent.mkdir()
    shutil.copyfile(run.script, spool)
    result, calls = run(dry=True, script=spool, changes={"DPLUS_REPO_ROOT": None})
    assert result.returncode == 0, result.stderr
    expected = "graph_settings=5 diffusion_settings=6" if run.analysis == "sensitivity" else "policies=27 sampling_S=2"
    assert f"dataset=pheme seed=41 task=0 {expected}" in result.stdout
    assert str(runtime / "Libs/configs/exp_manuscript_three_seed.yaml") in result.stdout
    assert result.stdout.index("--source-checkpoint") < result.stdout.index("Libs.exp.run") < result.stdout.index("--case")
    assert calls == []


def test_direct_execution_outside_slurm_is_rejected(launcher):
    run, *_ = launcher
    result, calls = run(slurm=False)
    assert result.returncode == 2 and "authorized Slurm job" in result.stderr and calls == []


@pytest.mark.parametrize("analysis", ("sensitivity", "threshold"))
def test_declared_resource_profile_is_retained(analysis):
    text = (SCRIPT_ROOT / f"run_manuscript_{analysis}.sh").read_text()
    for flag in ("--partition=cpu", "--nodelist=cpu04", "--cpus-per-task=8", "--mem=48G", "--array=0-11%1"):
        assert f"#SBATCH {flag}\n" in text
