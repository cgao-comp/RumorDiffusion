from __future__ import annotations

from Libs.exp.run import parse_seeds, read_yaml_like


def test_canonical_config_resolves_formal_seed_contract(monkeypatch):
    monkeypatch.setenv("RUMORDIFFUSION_PROCESSED_ROOT", "/tmp/rumordiffusion-processed")
    monkeypatch.setenv("RUMORDIFFUSION_TEXT_ENCODER", "/tmp/rumordiffusion-encoder")
    config = read_yaml_like("Libs/configs/exp_canonical.yaml")
    assert parse_seeds("", config) == [42, 3407, 126, 218, 91]
    assert config["dataset"]["mode"] == "eval_prefix"
    assert config["dataset"]["text_encoder"]["cache_name"] == "text_embs_mpnet_canonical_v1.pt"
    assert config["outputs"]["ckpt_dir"].startswith("Output/exploratory/")
    assert config["outputs"]["log_dir"].startswith("Output/exploratory/")
    assert config["outputs"]["results_dir"].startswith("Output/exploratory/")
    assert config["outputs"]["canonical"]["root"] == "Output"


def test_canonical_model_config_merges_manuscript_baseline_without_search_branches():
    config = read_yaml_like("Libs/configs/model_canonical/rumordiffusion.yaml")
    assert config["name"] == "rumordiffusion"
    assert config["model"]["arbiter"]["freeze_ctdm"] is True
    assert config["model"]["arbiter"]["posterior_aggregation"]["enabled"] is False
