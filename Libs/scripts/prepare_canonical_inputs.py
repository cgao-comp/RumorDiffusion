#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from Libs.data.dataset import GraphDataset
from Libs.data.processor import load_config, process_dataset, write_manifests_from_processed
from Libs.exp.run import ensure_text_embedding_cache, read_yaml_like, resolve_dataset_prefix_cfg, resolve_dataset_text_runtime_cfg
from Libs.utils.artifact_schema import file_sha256
from Libs.utils.provenance import write_json


def _label_mapping(dataset_dir: Path) -> dict[str, int]:
    import json

    payload = json.loads((dataset_dir / "label_schema.json").read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload["mapping"].items()}


def prepare(
    exp_path: str | Path,
    dataset_config_path: str | Path,
    *,
    datasets: list[str] | None = None,
    force_reprocess: bool = False,
) -> dict[str, Any]:
    exp_cfg = read_yaml_like(str(exp_path))
    processor_cfg = load_config(str(dataset_config_path))
    selected = datasets or [str(value) for value in exp_cfg.get("datasets", processor_cfg.datasets)]
    if not selected:
        raise ValueError("No datasets selected for canonical input preparation")
    text_runtime = dict(resolve_dataset_text_runtime_cfg(exp_cfg))
    text_runtime["raw_root"] = str(processor_cfg.raw_root)
    text_runtime["processed_root"] = str(processor_cfg.processed_root)
    dataset_cfg = exp_cfg.get("dataset", {}) if isinstance(exp_cfg.get("dataset", {}), dict) else {}
    prefix_base = dataset_cfg.get("prefix", {}) if isinstance(dataset_cfg.get("prefix", {}), dict) else {}
    processed_root = Path(str(processor_cfg.processed_root))
    report: dict[str, Any] = {
        "schema_version": "1",
        "exp_config": str(exp_path),
        "dataset_config": str(dataset_config_path),
        "datasets": {},
    }

    for dataset in selected:
        ensure_text_embedding_cache(dataset, text_runtime)
        dataset_dir = processed_root / dataset
        data_path = dataset_dir / "pyg" / "data.pt"
        slices_path = dataset_dir / "pyg" / "slices.pt"
        if force_reprocess or not data_path.is_file() or not slices_path.is_file():
            process_dataset(dataset, processor_cfg)
        write_manifests_from_processed(
            dataset,
            processor_cfg,
            generation_command=f"prepare_canonical_inputs.py --exp {exp_path} --dataset-config {dataset_config_path}",
        )
        prefix_cfg = resolve_dataset_prefix_cfg(prefix_base, dataset)
        label_schema = _label_mapping(dataset_dir)
        split_counts = {}
        for split in ("train", "val", "test"):
            loaded = GraphDataset(
                root=str(dataset_dir),
                split=split,
                mode=str(dataset_cfg.get("mode", "eval_prefix")),
                use_lp_edges=bool(dataset_cfg.get("use_lp_edges", False)),
                require_lp_edges=bool(dataset_cfg.get("require_lp_edges", False)),
                prefix_cfg=prefix_cfg,
                text_cache_name=str(text_runtime.get("cache_name", "text_embs.pt")),
                force_text_cache=bool(text_runtime.get("force_text_cache", False)),
                text_normalize=bool(text_runtime.get("normalize_embeddings", False)),
                label_schema=label_schema,
            )
            split_counts[split] = len(loaded)
        report["datasets"][dataset] = {
            "processed_root": str(dataset_dir),
            "text_cache": {
                "path": str(dataset_dir / str(text_runtime.get("cache_name", "text_embs.pt"))),
                "sha256": file_sha256(dataset_dir / str(text_runtime.get("cache_name", "text_embs.pt"))),
            },
            "manifests": {
                name: file_sha256(dataset_dir / name)
                for name in ("raw_manifest.json", "processed_manifest.json", "split_manifest.json", "label_schema.json")
            },
            "split_counts": split_counts,
        }
    output_root = Path(str(exp_cfg.get("outputs", {}).get("results_dir", "Output/exploratory/rumordiffusion_canonical_v1/legacy_csv")))
    report_path = output_root.parent / "preflight" / "canonical_input_prep_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare strict canonical input manifests and text caches.")
    parser.add_argument("--exp", default="Libs/configs/exp_canonical.yaml")
    parser.add_argument("--dataset-config", default="Libs/configs/dataset_canonical.yaml")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset subset; default uses exp/dataset config")
    parser.add_argument("--force-reprocess", action="store_true", help="Rebuild processed PyG data before writing manifests")
    args = parser.parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()] or None
    report = prepare(args.exp, args.dataset_config, datasets=datasets, force_reprocess=bool(args.force_reprocess))
    print(f"canonical input preparation passed: {report['report_path']}")


if __name__ == "__main__":
    main()
