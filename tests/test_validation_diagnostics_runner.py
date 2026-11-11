"""Exercise diagnostic scheduler wiring without scientific inference or job submission."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "Libs/scripts/run_validation_diagnostics.sh"


@pytest.fixture
def launcher(tmp_path):
    root, source, binary = (tmp_path / name for name in ("runtime", "source", "bin"))
    binary.mkdir()
    run_dir = root / "Output/canonical/rumordiffusion_canonical_v1/runs/run-5a8808d520d941fa111e"
    (run_dir / "checkpoints").mkdir(parents=True)
    for name in ("resolved_exp.yaml", "resolved_model.yaml", "checkpoints/best.pt"):
        (run_dir / name).write_text("disposable fixture")
    (source / "Libs/exp").mkdir(parents=True)
    (source / "Libs/exp/validation_diagnostics.py").write_text("# fixture")
    git = binary / "git"
    git.write_text("#!/usr/bin/env python3\nimport os,sys\nif 'rev-parse' in sys.argv: print(os.environ['FIXTURE_HEAD'])\nelif 'status' in sys.argv: print(os.environ.get('FIXTURE_DIRTY',''),end='')\n")
    python = binary / "fixture-python"
    python.write_text("""#!/usr/bin/env python3
import os,sys,json
args=sys.argv[1:]
mode='preflight' if '--check-inputs' in args else 'qualification' if '--check-only' in args else 'inference'
with open(os.environ['FIXTURE_LOG'],'a') as f:
 f.write(json.dumps({'mode':mode,'args':args,'cwd':os.getcwd(),'scratch':os.environ['TMPDIR'],
 'code':os.environ['DPLUS_VALIDATION_DIAGNOSTIC_CODE_COMMIT'],'threads':os.environ['OMP_NUM_THREADS'],
 'cuda':os.environ['CUDA_VISIBLE_DEVICES'],'pythonpath':os.environ['PYTHONPATH']})+'\\n')
if mode==os.environ.get('FIXTURE_FAIL'): sys.exit(29)
""")
    git.chmod(0o755)
    python.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DPLUS_VALIDATION_", "SLURM_"))}
    env.update(PATH=f"{binary}:{env['PATH']}", PYTHON_BIN=str(python), DPLUS_REPO_ROOT=str(root),
               DPLUS_VALIDATION_DIAGNOSTIC_SOURCE_ROOT=str(source), DPLUS_VALIDATION_DIAGNOSTIC_CODE_COMMIT="a" * 40,
               SLURM_JOB_ID="424242", SLURM_ARRAY_TASK_ID="0", SLURM_CPUS_PER_TASK="1", FIXTURE_HEAD="a" * 40,
               FIXTURE_LOG=str(log))

    def run(*args, changes=None, script=SCRIPT):
        current = dict(env)
        for k, v in (changes or {}).items():
            if v is None:
                current.pop(k, None)
            else:
                current[k] = str(v)
        if log.exists():
            log.unlink()
        result = subprocess.run(["bash", str(script), *args], cwd=root, env=current, text=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        assert all(not Path(call['scratch']).exists() for call in calls)
        return result, calls

    return run, root, source, run_dir, tmp_path


def test_complete_sequence_uses_preserved_inputs_and_exploratory_output(launcher):
    run, root, source, run_dir, _ = launcher
    result, calls = run()
    assert result.returncode == 0, result.stderr
    assert [x['mode'] for x in calls] == ['preflight', 'inference', 'qualification']
    args = calls[1]['args']
    assert args[args.index('--exp') + 1] == str(run_dir / 'resolved_exp.yaml')
    assert args[args.index('--ckpt') + 1] == str(run_dir / 'checkpoints/best.pt')
    assert args[args.index('--seeds') + 1] == '42'
    assert {'--validation_diagnostic_only', '--eval_only'} <= set(args)
    assert '--canonical_main_only' not in args and '--canonical_early_only' not in args
    output = Path(args[args.index('--diagnostic-output') + 1])
    assert output.is_relative_to(root / 'Output/exploratory')
    assert all(x['cwd'] == str(source) and x['pythonpath'] == str(source) for x in calls)
    assert all(x['threads'] == '1' and x['cuda'] == '' and x['code'] == 'a' * 40 for x in calls)
    assert calls[2]['args'][calls[2]['args'].index('--output') + 1] == str(output)


@pytest.mark.parametrize('mode,expected', [('preflight', ['preflight']), ('inference', ['preflight', 'inference']),
                                         ('qualification', ['preflight', 'inference', 'qualification'])])
def test_failure_stops_following_actions_and_preserves_exit(launcher, mode, expected):
    run, *_ = launcher
    result, calls = run(changes={'FIXTURE_FAIL': mode})
    assert result.returncode == 29
    assert [x['mode'] for x in calls] == expected


@pytest.mark.parametrize('changes', [
    {'DPLUS_VALIDATION_DIAGNOSTIC_CODE_COMMIT': None}, {'DPLUS_VALIDATION_DIAGNOSTIC_CODE_COMMIT': 'short'},
    {'FIXTURE_HEAD': 'b' * 40}, {'FIXTURE_DIRTY': ' M source.py'}, {'SLURM_ARRAY_TASK_ID': '1'},
    {'SLURM_ARRAY_TASK_ID': '00'}, {'SLURM_JOB_ID': None},
])
def test_invalid_pin_task_or_context_prevents_calls(launcher, changes):
    run, *_ = launcher
    result, calls = run(changes=changes)
    assert result.returncode == 2 and not calls


def test_spooled_dry_run_is_read_only_and_uses_submission_root(launcher):
    run, root, _, _, scratch = launcher
    spool = scratch / 'slurm_script'
    shutil.copyfile(SCRIPT, spool)
    result, calls = run('--dry-run', changes={'DPLUS_REPO_ROOT': None}, script=spool)
    assert result.returncode == 0 and not calls
    assert 'diagnostic_seeds=41,42,43 split=val hop=3' in result.stdout
    assert str(root / 'Output/canonical') in result.stdout
    assert result.stdout.index('--check-inputs') < result.stdout.index('Libs.exp.run') < result.stdout.index('--check-only')


def test_resource_profile_remains_authorized():
    text = SCRIPT.read_text()
    for value in ('--partition=cpu', '--nodelist=cpu04', '--cpus-per-task=8', '--mem=48G', '--array=0-0%1'):
        assert f'#SBATCH {value}\n' in text
