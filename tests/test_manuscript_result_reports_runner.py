"""Launcher fixtures exercise serialized report commands without Slurm or model calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "Libs/scripts/run_manuscript_result_reports.sh"
KINDS = ("early", "stopping", "case_study", "transfer", "sampling", "sensitivity", "threshold")


@pytest.fixture
def launcher(tmp_path):
    root, source, binary = (tmp_path / name for name in ("runtime", "source", "bin"))
    (source / "Libs/exp").mkdir(parents=True)
    binary.mkdir()
    (source / "Libs/exp/manuscript_result_reports.py").write_text("# fixture\n")
    declaration = root / "Output/exploratory/rumordiffusion_canonical_v1/control/manuscript_seed42_execution.json"
    declaration.parent.mkdir(parents=True)
    declaration.write_text("{}\n")
    git = binary / "git"
    git.write_text(
        '#!/usr/bin/env python3\nimport os,sys\nif "rev-parse" in sys.argv: print(os.environ["HEAD_FIXTURE"])\nif "status" in sys.argv: print(os.environ.get("DIRTY_FIXTURE",""),end="")\n'
    )
    git.chmod(0o755)
    python = binary / "fixture-python"
    python.write_text("""#!/usr/bin/env python3
import json,os,sys
with open(os.environ['CALLS_FIXTURE'],'w') as f:
    json.dump({'args':sys.argv[1:],'cwd':os.getcwd(),'scratch':os.environ['TMPDIR'],
               'cuda':os.environ['CUDA_VISIBLE_DEVICES'],'threads':os.environ['OMP_NUM_THREADS']},f)
sys.exit(int(os.environ.get('FAIL_FIXTURE','0')))
""")
    python.chmod(0o755)
    log = tmp_path / "calls.json"
    base = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "DPLUS_REPORT_"))}
    base.update(
        PATH=str(binary) + ":" + base["PATH"],
        PYTHON_BIN=str(python),
        DPLUS_REPO_ROOT=str(root),
        DPLUS_REPORT_SOURCE_ROOT=str(source),
        DPLUS_REPORT_CODE_COMMIT="a" * 40,
        HEAD_FIXTURE="a" * 40,
        CALLS_FIXTURE=str(log),
        SLURM_JOB_ID="fixture",
        SLURM_CPUS_PER_TASK="1",
    )

    def run(index=0, *, dry=False, changes=None):
        env = {**base, "SLURM_ARRAY_TASK_ID": str(index)}
        for key, value in (changes or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        if log.exists():
            log.unlink()
        result = subprocess.run(
            ["bash", str(SCRIPT)] + (["--dry-run"] if dry else []), env=env, cwd=root, capture_output=True, text=True
        )
        call = json.loads(log.read_text()) if log.exists() else None
        if call:
            assert not Path(call["scratch"]).exists()
        return result, call

    return run, root, source


@pytest.mark.parametrize("index", range(7))
def test_every_report_kind_uses_fixed_producers_and_the_selected_processing_source(launcher, index):
    run, root, source = launcher
    result, call = run(index)
    assert result.returncode == 0, result.stderr
    args = call["args"]
    assert args[args.index("--kind") + 1] == KINDS[index]
    assert args[args.index("--main-training-commit") + 1] == "80c76c3f59bd34e76d795af262d8d883bb83e0b5"
    assert args[args.index("--early-inference-commit") + 1] == "f36fc8a4b415c38635a922ca8a336e0eda1c9558"
    assert args[args.index("--sampling-inference-commit") + 1] == "94f1ad2ceb5935d5ca30b52346e136263e8e14a1"
    assert args[args.index("--source-repo") + 1] == str(source)
    assert ("--render-dir" in args) == (KINDS[index] == "case_study")
    assert call["cwd"] == str(source) and call["cuda"] == "" and call["threads"] == "1"


def test_dry_run_does_not_execute_and_exposes_actual_command(launcher):
    run, _, _ = launcher
    result, call = run(2, dry=True)
    assert result.returncode == 0 and call is None
    assert result.stdout.splitlines()[0] == "kind=case_study task=2 seeds=41,42,43"
    assert "--render-dir" in shlex.split(result.stdout.splitlines()[1])


@pytest.mark.parametrize(
    "changes",
    [
        {"SLURM_JOB_ID": None},
        {"DPLUS_REPORT_CODE_COMMIT": ""},
        {"HEAD_FIXTURE": "b" * 40},
        {"DIRTY_FIXTURE": " M source.py\n"},
    ],
)
def test_missing_job_or_incorrect_source_prevents_execution(launcher, changes):
    run, *_ = launcher
    result, call = run(changes=changes)
    assert result.returncode == 2 and call is None


@pytest.mark.parametrize("index", ["7", "-1", "01", "abc"])
def test_invalid_task_is_rejected(launcher, index):
    run, *_ = launcher
    result, call = run(index)
    assert result.returncode == 2 and call is None


def test_failed_gate_propagates_exit_and_cleans_scratch(launcher):
    run, *_ = launcher
    result, call = run(changes={"FAIL_FIXTURE": "17"})
    assert result.returncode == 17 and call is not None


def test_resource_directives_keep_serial_cpu_profile():
    text = SCRIPT.read_text()
    for flag in ("--partition=cpu", "--nodelist=cpu04", "--cpus-per-task=8", "--mem=48G", "--array=0-6%1"):
        assert "#SBATCH " + flag in text
