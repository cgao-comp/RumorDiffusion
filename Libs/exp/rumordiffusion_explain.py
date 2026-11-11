"""Interpretability and case-study runner for RumorDiffusion.

This script loads a RumorDiffusion checkpoint, runs inference on a chosen
split, and saves:
- per-sample logits, probabilities, and token-level statistics
- multi-window uncertainty traces for qualitative case studies
- automatically selected early-confident and deferred examples

All analysis parameters are resolved from YAML configs to avoid hardcoding.
"""

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

from Libs.data.dataset import GraphDataset
from Libs.exp.run import (apply_dataset_overrides, _deep_merge,
                          ensure_text_embedding_cache, load_checkpoint_state,
                          make_graph_dataset, read_yaml_like,
                          resolve_early_observation_spec,
                          resolve_dataset_text_runtime_cfg,
                          resolve_max_nodes_setting,
                          scan_max_nodes_across_splits,
                          validate_rumordiffusion_pair_config)
from Libs.models.baselines.CTDM_syn import CTDMSyn
from Libs.models.baselines.RumorDiffusion import RumorDiffusion
from Libs.models.modules.rumor_diffusion import _extract_source_ids_from_batch


def build_device(device_cfg: Dict) -> str:
    """Selects device string based on config and availability."""
    force_cpu = bool(device_cfg.get("force_cpu", False))
    if force_cpu:
        return "cpu"
    cfg_type = str(device_cfg.get("type", "cuda")).lower()
    if cfg_type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return cfg_type


def resolve_rd_cfg(exp_cfg: Dict, model_cfg: Dict, dataset_name: str, num_classes: int) -> Tuple[Dict, Dict]:
    """Mimics the main runner's merge order: model defaults, exp overrides."""
    rd_cfg_model = model_cfg.get("model", {}) if isinstance(model_cfg.get("model", {}), dict) else {}
    rd_cfg_exp = exp_cfg.get("rumordiffusion", {}) if isinstance(exp_cfg.get("rumordiffusion", {}), dict) else {}
    rd_cfg = _deep_merge(rd_cfg_model, rd_cfg_exp)
    rd_cfg = apply_dataset_overrides(rd_cfg, dataset_name)
    rd_cfg = validate_rumordiffusion_pair_config(rd_cfg, dataset_name, num_classes)
    rd_eval_model = model_cfg.get("evaluation", {}) if isinstance(model_cfg.get("evaluation", {}), dict) else {}
    rd_eval_exp = rd_cfg.get("evaluation", {}) if isinstance(rd_cfg.get("evaluation", {}), dict) else {}
    rd_eval_cfg = _deep_merge(rd_eval_model, rd_eval_exp)
    return rd_cfg, rd_eval_cfg


def resolve_ctdm_max_nodes(ds: GraphDataset, ctdm_cfg: Dict) -> int:
    """Resolves CTDM max-nodes budget from the dataset split and YAML config."""
    scanned_max_nodes = scan_max_nodes_across_splits([ds])
    configured_max_nodes = int(ctdm_cfg.get("max_nodes", 0) or 0)
    return resolve_max_nodes_setting(
        scanned_max_nodes=scanned_max_nodes,
        configured_max_nodes=configured_max_nodes,
        context="RumorDiffusion explain CTDM",
    )


def build_ctdm(ctdm_cfg: Dict, text_cfg: Dict, num_features: int, num_classes: int, max_nodes: int) -> CTDMSyn:
    """Constructs CTDMSyn with backbone, text, and regularizer settings."""
    bb_cfg = ctdm_cfg.get("backbone", {}) if isinstance(ctdm_cfg.get("backbone", {}), dict) else {}
    text_enabled = bool(text_cfg.get("enabled", False))
    text_dim = int(text_cfg.get("text_dim", 0)) if text_enabled else 0
    text_fuse = str(text_cfg.get("fuse_mode", "add")) if text_enabled else "add"

    model = CTDMSyn(
        in_channels=num_features,
        num_classes=num_classes,
        hidden_channels=int(ctdm_cfg.get("hidden", 128)),
        num_layers=int(ctdm_cfg.get("num_layers", 3)),
        dropout=float(ctdm_cfg.get("dropout", 0.3)),
        T=int(ctdm_cfg.get("T", 200)),
        obs_minutes=int(ctdm_cfg.get("obs_minutes", 30)),
        t_embed_dim=int(ctdm_cfg.get("t_embed_dim", ctdm_cfg.get("hidden", 128))),
        film_hidden=int(ctdm_cfg.get("film_hidden", 128)),
        max_nodes=int(max_nodes),
        backbone_name=str(bb_cfg.get("name", "gcn")).lower(),
        backbone_heads=int(bb_cfg.get("heads", 1)),
        backbone_dropout=float(bb_cfg.get("dropout", ctdm_cfg.get("dropout", 0.3))),
        deg_histogram=None,
        pna_aggregators=bb_cfg.get("aggregators", None),
        pna_scalers=bb_cfg.get("scalers", None),
        text_dim=(text_dim if text_enabled else None),
        text_fuse=text_fuse,
    )
    if "t_stride" in ctdm_cfg:
        setattr(model, "t_stride", int(ctdm_cfg["t_stride"]))
    if "parameterization" in ctdm_cfg:
        model.parameterization = str(ctdm_cfg["parameterization"]).lower()
    model.guidance_p = float(ctdm_cfg.get("guidance_p", 0.2))
    model.guidance_scale = float(ctdm_cfg.get("guidance_scale", 2.0))
    model.sampling_eta = float(ctdm_cfg.get("sampling_eta", 0.0))
    if "use_attn_readout" in ctdm_cfg:
        model.use_attn_readout = bool(ctdm_cfg["use_attn_readout"])

    mw_cfg = ctdm_cfg.get("multiwindow", {}) if isinstance(ctdm_cfg.get("multiwindow", {}), dict) else {}
    model.multiwindow_enabled = bool(mw_cfg.get("enabled", False))
    model.cutoff_min = int(mw_cfg.get("min", 10))
    model.cutoff_max = int(mw_cfg.get("max", 30))

    sp_cfg = ctdm_cfg.get("sparse_pairs", {}) if isinstance(ctdm_cfg.get("sparse_pairs", {}), dict) else {}
    model.sparse_pairs_enabled = bool(sp_cfg.get("enabled", False))
    model.sparse_pairs_strategy = str(sp_cfg.get("strategy", "b")).lower()
    model.sparse_pairs_topk = int(sp_cfg.get("topk", 8))
    if "lambda_time" in sp_cfg:
        model.sparse_pairs_lambda_time = float(sp_cfg["lambda_time"])
    if "lambda_degree" in sp_cfg:
        model.sparse_pairs_lambda_degree = float(sp_cfg["lambda_degree"])

    reg_cfg = ctdm_cfg.get("regularizers", {}) if isinstance(ctdm_cfg.get("regularizers", {}), dict) else {}
    if "lambda_proto" in reg_cfg:
        model.lambda_proto = float(reg_cfg["lambda_proto"])
    if "temperature" in reg_cfg:
        model.prototype_temperature = float(reg_cfg["temperature"])
    if "momentum" in reg_cfg:
        model.prototype_momentum = float(reg_cfg["momentum"])
    if "lambda_topo_bce" in reg_cfg:
        model.lambda_topo_bce = float(reg_cfg["lambda_topo_bce"])
    if "lambda_deg_kl" in reg_cfg:
        model.lambda_deg_kl = float(reg_cfg["lambda_deg_kl"])
    if "lambda_proto_orth" in reg_cfg:
        model.lambda_proto_orth = float(reg_cfg["lambda_proto_orth"])
    if "lambda_supcon_pair" in reg_cfg:
        model.lambda_supcon_pair = float(reg_cfg["lambda_supcon_pair"])
    if "pair_ids" in reg_cfg:
        model.pair_ids = tuple(reg_cfg["pair_ids"])
    return model


def infer_hidden_channels(ds: GraphDataset, ctdm: CTDMSyn, device: torch.device) -> Tuple[int, int]:
    """Infers potential and concatenated hidden dimensions from a dataset sample."""
    sample = ds.get(0)
    if hasattr(sample, "h0") and hasattr(sample, "h1"):
        h_dim = int(sample.h0.size(-1))
        concat_dim = int(torch.cat([sample.h0, sample.h1], dim=-1).size(-1))
        return h_dim, concat_dim
    loader = PyGDataLoader(ds, batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        potentials = ctdm.infer_potentials_all(batch)
    h_dim = int(potentials.size(-1))
    concat_dim = int(torch.cat([potentials[:, 0, :], potentials[:, 1, :]], dim=-1).size(-1))
    return h_dim, concat_dim


def build_rumor_diffusion(model_cfg: Dict, exp_cfg: Dict, ds: GraphDataset, device: torch.device) -> RumorDiffusion:
    """Builds RumorDiffusion model using YAML configuration."""
    num_features = ds.num_features
    num_classes = ds.num_classes
    dataset_name = os.path.basename(os.path.normpath(ds.root))
    rd_cfg, _rd_eval_cfg = resolve_rd_cfg(exp_cfg, model_cfg, dataset_name, int(num_classes))
    ctdm_cfg = rd_cfg.get("ctdm", {})
    arb_cfg = rd_cfg.get("arbiter", {})
    loss_cfg = rd_cfg.get("loss", {})
    dd_cfg = rd_cfg.get("dd_gan", {})
    text_cfg = rd_cfg.get("text", {})
    max_nodes = resolve_ctdm_max_nodes(ds, ctdm_cfg)
    ctdm = build_ctdm(ctdm_cfg, text_cfg, num_features, num_classes, max_nodes)
    ctdm = ctdm.to(device)

    hidden_dim, _ = infer_hidden_channels(ds, ctdm, device)

    model = RumorDiffusion(
        hidden_channels=hidden_dim,
        num_classes=num_classes,
        arbiter_hidden=int(arb_cfg.get("hidden", 256)),
        dropout=float(arb_cfg.get("dropout", 0.3)),
        ctdm=ctdm,
        freeze_ctdm=bool(arb_cfg.get("freeze_ctdm", True)),
        arbiter_type=str(arb_cfg.get("type", "transformer")).lower(),
        arbiter_readout=str(arb_cfg.get("readout", "cls")).lower(),
        token_norm=str(arb_cfg.get("token_norm", "layernorm")).lower(),
        token_dropout=float(arb_cfg.get("token_dropout", arb_cfg.get("dropout", 0.3))),
        transformer_nhead=int(arb_cfg.get("transformer", {}).get("nhead", 4)),
        transformer_nlayers=int(arb_cfg.get("transformer", {}).get("nlayers", 2)),
        transformer_dim_feedforward=int(arb_cfg.get("transformer", {}).get("dim_feedforward", 512)),
        transformer_dropout=float(arb_cfg.get("transformer", {}).get("dropout", arb_cfg.get("dropout", 0.3))),
        dd_gan_enabled=bool(dd_cfg.get("enabled", False)),
        enable_dd_gan=bool(dd_cfg.get("enable", dd_cfg.get("enabled", False))),
        lambda_gan=float(dd_cfg.get("lambda_gan", 0.0)),
        lr_d=float(dd_cfg.get("lr_d", 1.0e-4)),
        lambda_r1=float(dd_cfg.get("lambda_r1", 10.0)),
        use_pair_token=bool(arb_cfg.get("use_pair_token", False)),
        pair_token_mode=str(arb_cfg.get("pair_token_mode", "diff")),
        pair_ids=tuple(arb_cfg.get("pair_ids", [1, 2])),
        tf_head_enabled=bool(arb_cfg.get("tf_head_enabled", False)),
        tf_infer_adjust_alpha=float(arb_cfg.get("tf_infer_adjust_alpha", 0.0)),
        pair_margin_enabled=bool(loss_cfg.get("pair_margin_enabled", False)),
        pair_margin_m=float(loss_cfg.get("pair_margin_m", 0.5)),
        lambda_pair_margin=float(loss_cfg.get("lambda_pair_margin", 0.0)),
        enable_pair_margin_loss=bool(loss_cfg.get("enable_pair_margin_loss", True)),
        lambda_tf_bce=float(loss_cfg.get("lambda_tf_bce", 0.0)),
        use_class_embedding=bool(arb_cfg.get("use_class_embedding", False)),
        class_embed_mode=str(arb_cfg.get("class_embed_mode", "concat_proj")),
        cache_potentials=bool(arb_cfg.get("cache_potentials", False)),
        infer_amp=bool(arb_cfg.get("infer_amp", False)),
        uncertainty_force_ctdm=bool(arb_cfg.get("uncertainty_force_ctdm", False)),
        loss_logging=bool(loss_cfg.get("loss_logging", False)),
    )

    if "class_priors" in loss_cfg:
        pri = torch.tensor(loss_cfg["class_priors"], dtype=torch.float32, device=device)
        model.class_priors = pri / pri.sum()
    model.la_tau = float(loss_cfg.get("la_tau", 1.0))
    model.la_gamma = float(loss_cfg.get("la_gamma", 1.5))
    model.la_smooth = float(loss_cfg.get("la_smooth", 0.05))
    model.lambda_brier = float(loss_cfg.get("w_brier", 0.01))
    model.lambda_consistency = float(loss_cfg.get("w_cons", 0.05))

    mp_enabled = bool(loss_cfg.get("enable_multi_pair_margin", loss_cfg.get("multi_pair_margin_enabled", False)))
    model.multi_pair_margin_list = loss_cfg.get("multi_pair_margin_list", []) if mp_enabled else []
    model.multi_pair_margin_values = loss_cfg.get("multi_pair_margin_values", []) if mp_enabled else []
    model.lambda_multi_pair_margin = float(loss_cfg.get("lambda_multi_pair_margin", 0.0)) if mp_enabled else 0.0
    model.enable_multi_pair_margin = mp_enabled and len(model.multi_pair_margin_list) > 0 and float(model.lambda_multi_pair_margin) > 0.0

    model.enable_pair_margin_loss = bool(loss_cfg.get("enable_pair_margin_loss", loss_cfg.get("pair_margin_enabled", False)))
    if not model.enable_pair_margin_loss:
        model.lambda_pair_margin = 0.0
    model.enable_tf_head_loss = bool(loss_cfg.get("enable_tf_head_loss", True))
    if not model.enable_tf_head_loss:
        model.lambda_tf_bce = 0.0

    eval_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    if "class_priors" not in loss_cfg and isinstance(eval_cfg, dict) and "class_names" in eval_cfg:
        counts = [0 for _ in range(int(num_classes))]
        for raw_idx in range(len(ds)):
            yv = int(ds.get(raw_idx).y.view(-1)[0].item())
            counts[yv] += 1
        total = sum(counts)
        pri = torch.tensor([c / float(total) for c in counts], dtype=torch.float32, device=device)
        model.class_priors = pri
    return model.to(device)


def load_checkpoint(model: RumorDiffusion, ckpt_path: str, device: torch.device) -> None:
    """Loads model weights from checkpoint path."""
    state, _ = load_checkpoint_state(
        ckpt_path=ckpt_path,
        device=str(device),
        expected_metadata=None,
        validate_metadata=False,
        allow_legacy=True,
    )
    model.load_state_dict(state)


def parse_source_ids(raw_value: str) -> List[str]:
    """Parses a comma-separated source-id list."""
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def resolve_case_trace_cfg(dataset_name: str, rd_cfg: Dict, rd_eval_cfg: Dict, cli_source_ids: List[str]) -> Dict:
    """Builds the case-study trace configuration from YAML and CLI overrides."""
    explain_cfg = rd_cfg.get("explain", {}) if isinstance(rd_cfg.get("explain", {}), dict) else {}
    obs_minutes, observation_unit = resolve_early_observation_spec(
        eval_cfg={},
        model_eval_cfg=rd_eval_cfg,
        dataset_name=dataset_name,
    )
    if not obs_minutes:
        obs_minutes = [int(rd_cfg.get("ctdm", {}).get("obs_minutes", 30))]
    obs_minutes = [int(v) for v in obs_minutes]

    early_stop_cfg = rd_eval_cfg.get("early_stop", {}) if isinstance(rd_eval_cfg.get("early_stop", {}), dict) else {}
    early_stop_overrides = rd_eval_cfg.get("early_stop_overrides", {}) if isinstance(
        rd_eval_cfg.get("early_stop_overrides", {}), dict
    ) else {}
    if dataset_name in early_stop_overrides:
        early_stop_cfg = {**early_stop_cfg, **early_stop_overrides[dataset_name]}

    source_ids = cli_source_ids if cli_source_ids else parse_source_ids(explain_cfg.get("source_ids", ""))
    case_selection_cfg = explain_cfg.get("case_selection", {}) if isinstance(
        explain_cfg.get("case_selection", {}), dict
    ) else {}

    return {
        "enabled": bool(explain_cfg.get("enabled", True)),
        "obs_minutes": obs_minutes,
        "observation_unit": str(explain_cfg.get("observation_unit", observation_unit)),
        "samples": int(explain_cfg.get("samples", early_stop_cfg.get("samples", 2))),
        "tau_low": float(explain_cfg.get("tau_low", early_stop_cfg.get("tau_low", 0.3))),
        "tau_high": float(explain_cfg.get("tau_high", early_stop_cfg.get("tau_high", 0.7))),
        "gamma": float(explain_cfg.get("gamma", early_stop_cfg.get("gamma", 0.02))),
        "rumor_classes": list(explain_cfg.get("rumor_classes", early_stop_cfg.get("rumor_classes", [1, 2, 3]))),
        "source_ids": source_ids,
        "case_selection": {
            "enabled": bool(case_selection_cfg.get("enabled", True)),
        },
    }


def classify_decision(
    rumor_prob: float,
    rumor_var: float,
    tau_low: float,
    tau_high: float,
    gamma: float,
) -> Tuple[str, str]:
    """Applies the selective decision rule used in the main runner."""
    if rumor_var >= gamma:
        return "defer", "defer"
    if rumor_prob > tau_high:
        return "accept_rumor", "rumor"
    if rumor_prob < tau_low:
        return "accept_non_rumor", "non-rumor"
    return "defer", "defer"


def collect_case_traces(
    model: RumorDiffusion,
    loader: PyGDataLoader,
    device: torch.device,
    case_cfg: Dict,
    out_path: str,
) -> List[Dict]:
    """Collects multi-window prediction and uncertainty traces."""
    model.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    records: List[Dict] = []
    source_id_filter = set(case_cfg.get("source_ids", []))
    rumor_classes = [int(v) for v in case_cfg.get("rumor_classes", [1, 2, 3])]
    obs_minutes_list = [int(v) for v in case_cfg.get("obs_minutes", [])]
    samples = int(case_cfg.get("samples", 2))
    tau_low = float(case_cfg.get("tau_low", 0.3))
    tau_high = float(case_cfg.get("tau_high", 0.7))
    gamma = float(case_cfg.get("gamma", 0.02))
    obs_backup = int(model.ctdm.obs_minutes) if model.ctdm is not None else None

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            source_ids = _extract_source_ids_from_batch(batch)
            if len(source_ids) != 1:
                raise ValueError("Case-trace loader must use batch_size=1")
            source_id = str(source_ids[0])
            if source_id_filter and source_id not in source_id_filter:
                continue

            label = int(batch.y.view(-1)[0].item()) if hasattr(batch, "y") else -1
            label_binary = int(label in rumor_classes)
            traces: List[Dict] = []
            for obs_minutes in obs_minutes_list:
                if model.ctdm is not None:
                    model.ctdm.obs_minutes = int(obs_minutes)
                mean_prob, var_prob = model.predict_with_uncertainty(
                    batch,
                    num_samples=samples,
                    rumor_classes=None,
                )
                mean_prob = mean_prob[0]
                var_prob = var_prob[0]
                pred_class = int(torch.argmax(mean_prob).item())
                pred_prob = float(mean_prob[pred_class].item())
                pred_var = float(var_prob[pred_class].item())
                rumor_prob = float(mean_prob[rumor_classes].sum().item())
                rumor_var = float(var_prob[rumor_classes].sum().item())
                decision, accepted_label = classify_decision(
                    rumor_prob=rumor_prob,
                    rumor_var=rumor_var,
                    tau_low=tau_low,
                    tau_high=tau_high,
                    gamma=gamma,
                )
                accepted_binary = 1 if accepted_label == "rumor" else 0 if accepted_label == "non-rumor" else -1
                traces.append(
                    {
                        "obs_minutes": int(obs_minutes),
                        "pred_class": pred_class,
                        "pred_prob": pred_prob,
                        "pred_var": pred_var,
                        "rumor_prob": rumor_prob,
                        "rumor_var": rumor_var,
                        "decision": decision,
                        "accepted_label": accepted_label,
                        "accepted_binary": accepted_binary,
                        "label_binary": label_binary,
                        "decision_correct": int(accepted_binary == label_binary) if accepted_binary >= 0 else -1,
                        "probs": [float(v) for v in mean_prob.tolist()],
                        "vars": [float(v) for v in var_prob.tolist()],
                    }
                )

            earliest_accept = next((int(t["obs_minutes"]) for t in traces if t["decision"] != "defer"), None)
            records.append(
                {
                    "source_id": source_id,
                    "label": label,
                    "label_binary": label_binary,
                    "earliest_accept_minutes": earliest_accept,
                    "trace": traces,
                }
            )

    if obs_backup is not None and model.ctdm is not None:
        model.ctdm.obs_minutes = obs_backup

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def select_case_records(records: List[Dict]) -> Dict[str, Optional[Dict]]:
    """Selects one early-confident case and one deferred case."""
    if not records:
        return {"early_confident": None, "deferred": None}

    def first_trace(record: Dict) -> Dict:
        return record["trace"][0]

    early_candidates = [
        rec for rec in records
        if rec["trace"] and first_trace(rec)["decision"] != "defer" and first_trace(rec)["decision_correct"] == 1
    ]
    early_candidates.sort(
        key=lambda rec: (
            -first_trace(rec)["pred_prob"],
            first_trace(rec)["rumor_var"],
        )
    )

    deferred_candidates = [
        rec for rec in records
        if rec["trace"] and first_trace(rec)["decision"] == "defer" and any(t["decision"] != "defer" for t in rec["trace"][1:])
    ]
    if not deferred_candidates:
        deferred_candidates = [
            rec for rec in records if rec["trace"] and first_trace(rec)["decision"] == "defer"
        ]
    deferred_candidates.sort(
        key=lambda rec: (
            -sum(1 for t in rec["trace"] if t["decision"] == "defer"),
            -first_trace(rec)["rumor_var"],
        )
    )

    return {
        "early_confident": early_candidates[0] if early_candidates else None,
        "deferred": deferred_candidates[0] if deferred_candidates else None,
    }


def save_selected_cases(selected_cases: Dict[str, Optional[Dict]], out_path: str) -> None:
    """Saves selected case records as JSON."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        key: value for key, value in selected_cases.items() if value is not None
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def collect_explanations(
    model: RumorDiffusion,
    loader: PyGDataLoader,
    device: torch.device,
    pair_ids: Tuple[int, int],
    out_path: str,
) -> Tuple[int, float, float]:
    """Runs inference and saves JSON lines with logits and token stats."""
    model.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    records: List[Dict] = []
    p0, p1 = int(pair_ids[0]), int(pair_ids[1])
    total_prob = 0.0
    total_margin = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            tokens = model._build_tokens_from_batch(batch)
            logits, cls_out = model._encode_from_tokens(tokens)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = probs.max(dim=1)
            margin = logits[:, p0] - logits[:, p1]
            cls_norm = cls_out.norm(dim=1)
            token_norm_mean = tokens.norm(dim=2).mean(dim=1)
            token_norm_max = tokens.norm(dim=2).max(dim=1).values
            sids = _extract_source_ids_from_batch(batch)

            labels = batch.y.view(-1).detach().cpu() if hasattr(batch, "y") else None
            for i, sid in enumerate(sids):
                rec = {
                    "source_id": sid,
                    "pred_class": int(top_idx[i].item()),
                    "pred_prob": float(top_prob[i].item()),
                    "pair_margin": float(margin[i].item()),
                    "prob_true": float(probs[i, p0].item()) if p0 < probs.size(1) else 0.0,
                    "prob_false": float(probs[i, p1].item()) if p1 < probs.size(1) else 0.0,
                    "cls_norm": float(cls_norm[i].item()),
                    "token_norm_mean": float(token_norm_mean[i].item()),
                    "token_norm_max": float(token_norm_max[i].item()),
                    "probs": [float(v) for v in probs[i].tolist()],
                }
                if labels is not None and i < labels.numel():
                    rec["label"] = int(labels[i].item())
                    rec["correct"] = int(rec["pred_class"] == rec["label"])
                records.append(rec)
                total_prob += float(top_prob[i].item())
                total_margin += float(margin[i].item())

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    count = len(records)
    mean_prob = total_prob / max(1, count)
    mean_margin = total_margin / max(1, count)
    return count, mean_prob, mean_margin


def append_results_csv(csv_path: str, headers: List[str], row: List[str]) -> None:
    """Appends interpretability summary to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(headers)
        w.writerow(row)


def main() -> None:
    """CLI entrypoint for RumorDiffusion interpretability export."""
    parser = argparse.ArgumentParser(description="RumorDiffusion interpretability runner")
    parser.add_argument("--exp", type=str, required=True, help="Path to exp.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to model yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path for RumorDiffusion")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to explain")
    parser.add_argument("--out", type=str, default="Output/explain/rumordiffusion_explain.jsonl", help="Output JSONL path")
    parser.add_argument("--trace_out", type=str, default="", help="Optional JSONL path for multi-window case traces")
    parser.add_argument("--selected_out", type=str, default="", help="Optional JSON path for selected case-study examples")
    parser.add_argument("--source_ids", type=str, default="", help="Optional comma-separated source ids for targeted case tracing")
    args = parser.parse_args()

    exp_cfg = read_yaml_like(args.exp)
    model_cfg = read_yaml_like(args.model)
    device_cfg = exp_cfg.get("device", {})
    device_str = build_device(device_cfg)
    device = torch.device(device_str)
    if device_str == "cuda" and torch.cuda.is_available():
        gpu_id = int(device_cfg.get("gpu_id", 0))
        torch.cuda.set_device(gpu_id)

    dataset_cfg = exp_cfg["dataset"]
    dataset_name = dataset_cfg["name"]
    dataset_root = os.path.join(dataset_cfg["root"], dataset_name)
    use_lp_edges = bool(dataset_cfg.get("use_lp_edges", False))
    prefix_cfg = dataset_cfg.get("prefix", {})
    observability_cfg = dataset_cfg.get("observability", {})
    text_runtime_cfg = resolve_dataset_text_runtime_cfg(exp_cfg)
    ensure_text_embedding_cache(dataset_name, text_runtime_cfg)
    ds = make_graph_dataset(
        root=dataset_root,
        split=args.split,
        use_lp_edges=use_lp_edges,
        prefix_cfg=prefix_cfg,
        observability_cfg=observability_cfg,
        text_runtime_cfg=text_runtime_cfg,
    )
    rd_cfg, rd_eval_cfg = resolve_rd_cfg(exp_cfg, model_cfg, dataset_name, int(ds.num_classes))
    eval_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    train_cfg = exp_cfg.get("training", {}) if isinstance(exp_cfg.get("training", {}), dict) else {}
    max_samples = int(eval_cfg.get("max_samples", train_cfg.get("max_samples", 0)))
    ds_eval = ds if max_samples <= 0 else Subset(ds, list(range(min(len(ds), max_samples))))
    outputs_cfg = exp_cfg.get("outputs", {})
    results_dir = outputs_cfg.get("results_dir", os.path.join("Output", "results"))

    model = build_rumor_diffusion(model_cfg, exp_cfg, ds, device)
    load_checkpoint(model, args.ckpt, device)

    batch_size = int(exp_cfg.get("training", {}).get("batch_size", 4))
    loader = PyGDataLoader(
        ds_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(exp_cfg.get("training", {}).get("num_workers", 0)),
        pin_memory=(device_str == "cuda"),
        persistent_workers=int(exp_cfg.get("training", {}).get("num_workers", 0)) > 0,
    )

    arb_cfg = rd_cfg.get("arbiter", {})
    pair_ids = tuple(arb_cfg.get("pair_ids", [1, 2]))

    count, mean_prob, mean_margin = collect_explanations(model, loader, device, pair_ids, args.out)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_headers = [
        "timestamp",
        "dataset",
        "split",
        "model_cfg",
        "exp_cfg",
        "ckpt_path",
        "out_jsonl",
        "num_records",
        "mean_pred_prob",
        "mean_pair_margin",
        "pair_ids",
    ]
    summary_row = [
        ts,
        dataset_name,
        args.split,
        args.model,
        args.exp,
        args.ckpt,
        args.out,
        str(count),
        f"{mean_prob:.6f}",
        f"{mean_margin:.6f}",
        str(list(pair_ids)),
    ]
    summary_csv = os.path.join(results_dir, f"{dataset_name}_explain.csv")
    append_results_csv(summary_csv, summary_headers, summary_row)

    cli_source_ids = parse_source_ids(args.source_ids)
    case_cfg = resolve_case_trace_cfg(dataset_name, rd_cfg, rd_eval_cfg, cli_source_ids)
    if bool(case_cfg.get("enabled", True)):
        trace_out = args.trace_out or os.path.join(results_dir, f"{dataset_name}_case_traces.jsonl")
        case_loader = PyGDataLoader(
            ds_eval,
            batch_size=1,
            shuffle=False,
            num_workers=int(exp_cfg.get("training", {}).get("num_workers", 0)),
            pin_memory=(device_str == "cuda"),
            persistent_workers=int(exp_cfg.get("training", {}).get("num_workers", 0)) > 0,
        )
        trace_records = collect_case_traces(
            model=model,
            loader=case_loader,
            device=device,
            case_cfg=case_cfg,
            out_path=trace_out,
        )
        if bool(case_cfg.get("case_selection", {}).get("enabled", True)):
            selected_out = args.selected_out or os.path.join(results_dir, f"{dataset_name}_selected_cases.json")
            selected_cases = select_case_records(trace_records)
            save_selected_cases(selected_cases, selected_out)
    print(f"Saved interpretability records to {args.out}")


if __name__ == "__main__":
    main()
