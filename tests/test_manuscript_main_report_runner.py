"""Exercise report orchestration with disposable commands; never submit jobs."""

from pathlib import Path
import shutil

import pytest

from test_manuscript_main_runner import launcher as launcher_fixture

launcher = launcher_fixture
SCRIPT = Path(__file__).resolve().parents[1] / "Libs/scripts/run_manuscript_main_report.sh"
RECOVERY = "Libs.scripts.recover_lp_main_publication"
PAIRED = "Libs.exp.manuscript_main_comparison"


@pytest.fixture
def report_runner(launcher):
    run, runtime, source, scratch = launcher
    declaration = runtime / "Output/exploratory/rumordiffusion_canonical_v1/control/manuscript_seed42_execution.json"
    declaration.parent.mkdir(parents=True)
    declaration.write_text("{}")
    (source / "Libs/scripts").mkdir(parents=True, exist_ok=True)
    (source / "Libs/scripts/recover_lp_main_publication.py").write_text("# fixture only\n")
    fake_python = scratch / "bin/fixture-python"
    with fake_python.open("a") as stream:
        stream.write(
            "\nphase = 'recovery' if '--recover' in sys.argv else 'main' if '--bind-main' in sys.argv else 'render' if '--render-dir' in sys.argv else 'paired'\nif phase == os.environ.get('FIXTURE_FAIL_PHASE'): sys.exit(17)\n"
        )
    defaults = {"DPLUS_MAIN_COMPARISON_SOURCE_ROOT": source, "DPLUS_MAIN_COMPARISON_CODE_COMMIT": "a" * 40}

    def invoke(*, changes=None, **kwargs):
        return run(early=False, script=kwargs.pop("script", SCRIPT), changes={**defaults, **(changes or {})}, **kwargs)

    return invoke, runtime, source, scratch


def test_fresh_report_order_pins_and_complete_baseline_family(report_runner):
    run, runtime, source, _ = report_runner
    result, calls = run()
    assert result.returncode == 0, result.stderr
    assert [call["module"] for call in calls] == [RECOVERY, PAIRED, PAIRED, PAIRED]
    for call in calls[1:]:
        args = call["args"]
        assert args.count("--model-training-code-commit") == 19
        assert args[args.index("--training-code-commit") + 1] == "80c76c3f59bd34e76d795af262d8d883bb83e0b5"
        assert "temporalgcn=f36fc8a4b415c38635a922ca8a336e0eda1c9558" in args
        assert str(runtime / "Libs/configs/exp_manuscript_three_seed.yaml") in args
        assert call["cwd"] == call["pythonpath"] == str(source)
        assert call["cuda"] == "" and call["threads"] == "1"
    assert "--bind-main" in calls[1]["args"]
    assert "--render-dir" in calls[-1]["args"]


def test_existing_artifacts_are_verified_instead_of_rewritten(report_runner):
    run, runtime, _, _ = report_runner
    aggregate = runtime / "Output/canonical/rumordiffusion_canonical_v1/aggregates"
    aggregate.mkdir(parents=True)
    for name in ("main_results", "main_registry", "paired_results", "paired_registry"):
        (aggregate / f"manuscript_{name}.json").write_text("preserved")
    fragments = runtime / "Output/exploratory/rumordiffusion_canonical_v1/reports/manuscript_main_tables"
    fragments.mkdir(parents=True)
    for name in ("main", "paired"):
        (fragments / f"manuscript_{name}.tex").write_text("preserved")
    result, calls = run()
    assert result.returncode == 0, result.stderr
    assert [call["module"] for call in calls] == [RECOVERY, PAIRED, PAIRED]
    assert "--check-only" in calls[1]["args"]
    assert "--validate-fragments" in calls[2]["args"]
    assert all(path.read_text() == "preserved" for path in aggregate.iterdir())
    assert all(path.read_text() == "preserved" for path in fragments.iterdir())


@pytest.mark.parametrize("phase,expected", [("main", 2), ("paired", 3), ("render", 4)])
def test_failure_stops_downstream_steps_and_cleans_scratch(report_runner, phase, expected):
    run, _, _, _ = report_runner
    result, calls = run(changes={"FIXTURE_FAIL_PHASE": phase})
    assert result.returncode == 17
    assert len(calls) == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"DPLUS_MAIN_COMPARISON_CODE_COMMIT": None},
        {"DPLUS_MAIN_COMPARISON_CODE_COMMIT": "b" * 40},
        {"FIXTURE_DIRTY": " M source.py"},
    ],
)
def test_missing_mismatched_or_dirty_source_stops_before_python(report_runner, changes):
    result, calls = report_runner[0](changes=changes)
    assert result.returncode == 2 and not calls


def test_requires_slurm_and_single_array_task(report_runner):
    result, calls = report_runner[0](slurm=False)
    assert result.returncode == 2 and not calls
    result, calls = report_runner[0](index=1)
    assert result.returncode == 2 and not calls


def test_dry_run_and_spool_paths(report_runner):
    run, _, _, scratch = report_runner
    result, calls = run(dry=True, slurm=False)
    assert result.returncode == 0 and not calls
    assert "240_main_runs,76_pairs,152_metric_comparisons" in result.stdout
    spool = scratch / "spooled-script"
    shutil.copyfile(SCRIPT, spool)
    result, calls = run(script=spool, changes={"DPLUS_REPO_ROOT": None})
    assert result.returncode == 0 and len(calls) == 4
