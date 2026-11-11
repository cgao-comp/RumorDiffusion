"""Exercise the manuscript launcher with disposable tools and no scientific jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "Libs/scripts/run_manuscript_main.sh"
MAIN_GATE = "Libs.exp.manuscript_main_evidence"
EARLY_GATE = "Libs.exp.manuscript_early_evidence"
INFERENCE = "Libs.exp.run"
BASELINES = ("textonly", "bigcn", "tgat", "lp_gcn", "dga_fake", "agad", "hpcl_cascade", "tpso", "gcn", "gat",
             "gatv2", "sage", "appnp", "gcn2", "resgcn", "root_pool", "temporal_gcn", "ctdm_syn", "cgvae_gcn")


@pytest.fixture
def launcher(tmp_path):
    runtime = tmp_path / "runtime"
    source = tmp_path / "source"
    binary = tmp_path / "bin"
    binary.mkdir()
    (runtime / "Libs/configs").mkdir(parents=True)
    (runtime / "Libs/configs/exp_manuscript_three_seed.yaml").write_text("{}\n")
    (source / "Libs/configs/model_canonical").mkdir(parents=True)
    (source / "Libs/configs/model_canonical/rumordiffusion.yaml").write_text("{}\n")
    (source / "Libs/configs/baselines").mkdir()
    for name in BASELINES:
        (source / f"Libs/configs/baselines/{name}.yaml").write_text("{}\n")
    (source / "Libs/exp").mkdir()
    (source / "Libs/exp/manuscript_early_evidence.py").write_text("# fixture only\n")
    (source / "Libs/exp/manuscript_main_evidence.py").write_text("# fixture only\n")
    log = tmp_path / "calls.jsonl"
    git = binary / "git"
    git.write_text("""#!/usr/bin/env python3
import os, sys
if "rev-parse" in sys.argv:
    print(os.environ["FIXTURE_HEAD"])
elif "status" in sys.argv:
    print(os.environ.get("FIXTURE_DIRTY", ""), end="")
""")
    python = binary / "fixture-python"
    python.write_text("""#!/usr/bin/env python3
import json, os, sys
module = sys.argv[sys.argv.index("-m") + 1]
with open(os.environ["FIXTURE_CALL_LOG"], "a") as stream:
    stream.write(json.dumps({"module": module, "args": sys.argv[1:], "cwd": os.getcwd(),
        "tmpdir": os.environ["TMPDIR"], "pythonpath": os.environ["PYTHONPATH"],
        "cuda": os.environ["CUDA_VISIBLE_DEVICES"], "threads": os.environ["OMP_NUM_THREADS"],
        "output_root": os.environ["DPLUS_OUTPUT_ROOT"]}) + "\\n")
if module == os.environ.get("FIXTURE_FAIL_MODULE"):
    sys.exit(int(os.environ["FIXTURE_FAIL_CODE"]))
""")
    git.chmod(0o755)
    python.chmod(0o755)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("DPLUS_MANUSCRIPT_", "SLURM_"))}
    environment.update(
        PATH=f"{binary}:{environment['PATH']}", PYTHON_BIN=str(python),
        DPLUS_REPO_ROOT=str(runtime), DPLUS_MANUSCRIPT_SOURCE_ROOT=str(source),
        DPLUS_MANUSCRIPT_CODE_COMMIT="a" * 40,
        DPLUS_MANUSCRIPT_BASELINE_SOURCE_ROOT=str(source), DPLUS_MANUSCRIPT_BASELINE_CODE_COMMIT="a" * 40,
        DPLUS_MANUSCRIPT_QUALIFICATION_SOURCE_ROOT=str(source), DPLUS_MANUSCRIPT_QUALIFICATION_CODE_COMMIT="a" * 40,
        DPLUS_MANUSCRIPT_EXECUTION_PROVENANCE=str(runtime / "Libs/configs/exp_manuscript_three_seed.yaml"),
        DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT="b" * 40,
        DPLUS_MANUSCRIPT_BASELINE_TRAINING_CODE_COMMIT="c" * 40,
        FIXTURE_HEAD="a" * 40, FIXTURE_CALL_LOG=str(log), SLURM_CPUS_PER_TASK="1",
    )

    def run(index=0, *, early=True, dry=False, slurm=True, changes=None, script=SCRIPT, args=()):
        env = dict(environment, SLURM_ARRAY_TASK_ID=str(index))
        if slurm:
            env["SLURM_JOB_ID"] = "424242"
        for key, value in (changes or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
        if log.exists():
            log.unlink()
        command = ["bash", str(script)]
        if early:
            command.append("--early-evaluation")
        if dry:
            command.append("--dry-run")
        result = subprocess.run(command + list(args), env=env, cwd=runtime, text=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        assert all(not Path(call["tmpdir"]).exists() for call in calls), "launcher scratch was not cleaned"
        return result, calls

    return run, runtime, source, tmp_path


@pytest.mark.parametrize("index,dataset,model,seed", [
    (0, "pheme", "rumordiffusion", 41), (2, "pheme", "rumordiffusion", 43),
    (3, "pheme", "textonly", 41), (51, "pheme", "temporalgcn", 41),
    (60, "weibo", "rumordiffusion", 41), (120, "twitter15", "rumordiffusion", 41),
    (180, "twitter16", "rumordiffusion", 41), (239, "twitter16", "cgvae_gcn", 43),
])
def test_early_main_inference_and_three_horizon_gate_order(launcher, index, dataset, model, seed):
    run, runtime, source, _ = launcher
    result, calls = run(index)
    assert result.returncode == 0, result.stderr
    assert [call["module"] for call in calls] == [MAIN_GATE, INFERENCE, EARLY_GATE]
    for call in (calls[0], calls[2]):
        args = call["args"]
        assert args[args.index("--case") + 1:args.index("--case") + 4] == [dataset, model, str(seed)]
        assert args[args.index("--training-code-commit") + 1] == "b" * 40
        if model == "rumordiffusion":
            assert "--model-training-code-commit" not in args
        else:
            assert args[args.index("--model-training-code-commit") + 1] == f"{model}={'c' * 40}"
    args = calls[2]["args"]
    assert args[args.index("--inference-code-commit") + 1] == "a" * 40
    assert {"--eval_only", "--canonical_early_only", "--canonical_artifacts_only"} <= set(calls[1]["args"])
    assert "--canonical_main_only" not in calls[1]["args"]
    assert all(call["cwd"] == str(source) and call["pythonpath"] == str(source) for call in calls)
    assert all(call["cuda"] == "" and call["threads"] == "1" for call in calls)
    assert all(call["output_root"] == str(runtime / "Output") for call in calls)


def test_baseline_inference_pin_is_distinct_from_both_training_pins(launcher):
    run, _, source, scratch = launcher
    baseline_source = scratch / "baseline-source"
    shutil.copytree(source, baseline_source)
    result, calls = run(5, changes={
        "DPLUS_MANUSCRIPT_BASELINE_SOURCE_ROOT": baseline_source,
        "DPLUS_MANUSCRIPT_BASELINE_CODE_COMMIT": "d" * 40, "FIXTURE_HEAD": "d" * 40, "DPLUS_MANUSCRIPT_QUALIFICATION_CODE_COMMIT": "d" * 40,
    })
    assert result.returncode == 0, result.stderr
    args = calls[-1]["args"]
    assert args[args.index("--inference-code-commit") + 1] == "d" * 40
    assert args[args.index("--training-code-commit") + 1] == "b" * 40
    assert args[args.index("--model-training-code-commit") + 1] == f"textonly={'c' * 40}"
    assert calls[0 if len(calls) == 2 else 1]["cwd"] == str(baseline_source)
    assert calls[-1]["cwd"] == str(source)


@pytest.mark.parametrize("module,code,expected", [
    (MAIN_GATE, 17, [MAIN_GATE]),
    (INFERENCE, 18, [MAIN_GATE, INFERENCE]),
    (EARLY_GATE, 19, [MAIN_GATE, INFERENCE, EARLY_GATE]),
])
def test_failure_exit_and_cleanup(launcher, module, code, expected):
    run, _, _, _ = launcher
    result, calls = run(changes={"FIXTURE_FAIL_MODULE": module, "FIXTURE_FAIL_CODE": code})
    assert result.returncode == code
    assert [call["module"] for call in calls] == expected


@pytest.mark.parametrize("index,changes,message", [
    (0, {"DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT": None}, "DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT"),
    (5, {"DPLUS_MANUSCRIPT_BASELINE_TRAINING_CODE_COMMIT": None}, "DPLUS_MANUSCRIPT_BASELINE_TRAINING_CODE_COMMIT"),
    (0, {"DPLUS_MANUSCRIPT_CODE_COMMIT": None}, "approved clean source"),
    (0, {"FIXTURE_HEAD": "e" * 40}, "qualification source"),
    (0, {"FIXTURE_DIRTY": " M model.py\n"}, "qualification source"),
])
def test_missing_or_conflicting_provenance_prevents_execution(launcher, index, changes, message):
    run, _, _, _ = launcher
    result, calls = run(index, changes=changes)
    assert result.returncode == 2
    assert message in result.stderr
    assert calls == []


def test_missing_early_gate_prevents_inference(launcher):
    run, _, source, _ = launcher
    (source / "Libs/exp/manuscript_early_evidence.py").unlink()
    result, calls = run()
    assert result.returncode == 2
    assert "requires a source checkout" in result.stderr
    assert calls == []


def test_main_training_path_does_not_require_early_gate(launcher):
    run, _, source, _ = launcher
    (source / "Libs/exp/manuscript_early_evidence.py").unlink()
    result, calls = run(early=False, changes={"DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT": None})
    assert result.returncode == 0, result.stderr
    assert [call["module"] for call in calls] == [INFERENCE, MAIN_GATE]
    assert calls[1]["args"][calls[1]["args"].index("--training-code-commit") + 1] == "a" * 40
    assert "--canonical_main_only" in calls[0]["args"]
    assert "--eval_only" not in calls[0]["args"]


def test_dry_run_and_spooled_script_preserve_runtime_root(launcher):
    run, runtime, _, scratch = launcher
    spool_script = scratch / "spool" / "slurm_script"
    spool_script.parent.mkdir()
    shutil.copyfile(SCRIPT, spool_script)
    result, calls = run(dry=True, script=spool_script, changes={"DPLUS_REPO_ROOT": None})
    assert result.returncode == 0, result.stderr
    assert "dataset=pheme model=rumordiffusion seed=41 task=0" in result.stdout
    assert result.stdout.index(MAIN_GATE) < result.stdout.index(f"-m {INFERENCE} ") < result.stdout.index(EARLY_GATE)
    assert str(runtime / "Libs/configs/exp_manuscript_three_seed.yaml") in result.stdout
    assert calls == []


def test_execution_outside_slurm_is_rejected(launcher):
    run, _, _, _ = launcher
    result, calls = run(slurm=False)
    assert result.returncode == 2
    assert "authorized Slurm job" in result.stderr
    assert calls == []


@pytest.mark.parametrize("index", ["-1", "240", "01", "word"])
def test_undeclared_task_is_rejected(launcher, index):
    run, _, _, _ = launcher
    result, calls = run(index)
    assert result.returncode == 2
    assert "Invalid manuscript main task" in result.stderr
    assert calls == []


def test_authorized_resource_profile_is_preserved():
    script = SCRIPT.read_text()
    for directive in ("--partition=cpu", "--nodelist=cpu04", "--cpus-per-task=8", "--mem=48G", "--array=0-239%1"):
        assert f"#SBATCH {directive}\n" in script


@pytest.mark.parametrize("position,selector", list(enumerate(BASELINES, 1)))
def test_every_baseline_main_case_qualifies_with_its_actual_training_source(launcher, position, selector):
    run, _, source, scratch = launcher
    baseline_source = scratch / "baseline-source"
    shutil.copytree(source, baseline_source)
    result, calls = run(position * 3, early=False, changes={
        "DPLUS_MANUSCRIPT_BASELINE_SOURCE_ROOT": baseline_source,
        "DPLUS_MANUSCRIPT_BASELINE_CODE_COMMIT": "d" * 40, "FIXTURE_HEAD": "d" * 40, "DPLUS_MANUSCRIPT_QUALIFICATION_CODE_COMMIT": "d" * 40,
    })
    assert result.returncode == 0, result.stderr
    assert [call["module"] for call in calls] == [INFERENCE, MAIN_GATE]
    args = calls[1]["args"]
    model = "temporalgcn" if selector == "temporal_gcn" else selector
    assert args[args.index("--case") + 1:args.index("--case") + 4] == ["pheme", model, "41"]
    assert args[args.index("--training-code-commit") + 1] == "d" * 40
    assert calls[0 if len(calls) == 2 else 1]["cwd"] == str(baseline_source)
    assert calls[-1]["cwd"] == str(source)


@pytest.mark.parametrize("module,code,expected", [(INFERENCE, 23, [INFERENCE]), (MAIN_GATE, 24, [INFERENCE, MAIN_GATE])])
def test_main_training_or_qualification_failure_propagates(launcher, module, code, expected):
    run, *_ = launcher
    result, calls = run(early=False, changes={"FIXTURE_FAIL_MODULE": module, "FIXTURE_FAIL_CODE": code})
    assert result.returncode == code
    assert [call["module"] for call in calls] == expected


def test_missing_main_gate_prevents_training(launcher):
    run, _, source, _ = launcher
    (source / "Libs/exp/manuscript_main_evidence.py").unlink()
    result, calls = run(early=False)
    assert result.returncode == 2 and not calls
    assert "requires a source checkout" in result.stderr


def test_main_dry_run_shows_training_before_qualification(launcher):
    run, *_ = launcher
    result, calls = run(early=False, dry=True)
    assert result.returncode == 0 and not calls
    assert result.stdout.index(f"-m {INFERENCE} ") < result.stdout.index(MAIN_GATE)


@pytest.mark.parametrize("early,expected", [(False, [MAIN_GATE]), (True, [MAIN_GATE, EARLY_GATE])])
def test_qualified_pheme_seed42_is_never_reexecuted(launcher, early, expected):
    run, *_ = launcher
    result, calls = run(1, early=early)
    assert result.returncode == 0, result.stderr
    assert [call["module"] for call in calls] == expected
