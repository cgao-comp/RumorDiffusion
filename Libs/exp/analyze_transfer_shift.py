"""Transfer shift analysis for RumorDiffusion source/target dataset pairs."""

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import torch

from Libs.data.processor import (collect_json_source_texts, normalize_tweet_text,
                                 parse_source_texts)
from Libs.exp.run import (ensure_text_embedding_cache, make_graph_dataset,
                          read_yaml_like, resolve_dataset_text_runtime_cfg)


def load_text_mapping(dataset_name: str, raw_root: str) -> Dict[str, str]:
    """Loads source text mapping for one dataset."""
    raw_dir = os.path.join(raw_root, dataset_name)
    if dataset_name in {"weibo", "pheme"}:
        return collect_json_source_texts(raw_dir)
    return parse_source_texts(raw_dir)


def tokenize(text: str) -> List[str]:
    """Returns a lightweight tokenization for report statistics."""
    return re.findall(r"[a-z0-9_<>]+", str(text).lower())


def iter_split_samples(ds) -> Iterable[Tuple[str, torch.Tensor]]:
    """Yields (source_id, label) pairs from a GraphDataset split."""
    for idx in range(len(ds)):
        sample = ds.get(idx)
        yield str(sample.source_id), sample.y.view(-1).detach().cpu()


def summarize_split(ds, raw_text_map: Dict[str, str], normalized_text_map: Dict[str, str]) -> Dict[str, Any]:
    """Summarizes graph and text statistics for one split."""
    labels = Counter()
    token_lengths: List[int] = []
    node_counts: List[int] = []
    edge_counts: List[int] = []
    depths: List[float] = []
    time_spans: List[float] = []
    url_count = 0
    mention_count = 0
    hashtag_count = 0
    sample_count = 0
    for idx in range(len(ds)):
        sample = ds.get(idx)
        sid = str(sample.source_id)
        raw_text = raw_text_map.get(sid, "")
        normalized_text = normalized_text_map.get(sid, "")
        labels[int(sample.y.view(-1)[0].item())] += 1
        token_lengths.append(len(tokenize(normalized_text)))
        node_counts.append(int(sample.num_nodes))
        edge_counts.append(int(sample.edge_index.size(1)))
        depth_value = float(sample.x[:, 1].max().item()) if sample.x.dim() == 2 and int(sample.x.size(1)) > 1 else 0.0
        depths.append(depth_value)
        time_spans.append(float(sample.node_time_minutes_shifted.max().item()) if hasattr(sample, "node_time_minutes_shifted") else 0.0)
        url_count += int(bool(re.search(r"https?://\S+|www\.\S+", raw_text)))
        mention_count += int(bool(re.search(r"@\w+", raw_text)))
        hashtag_count += int(bool(re.search(r"#\w+", raw_text)))
        sample_count += 1
    denom = max(1, sample_count)
    return {
        "num_samples": sample_count,
        "label_counts": dict(sorted(labels.items())),
        "mean_text_tokens": float(sum(token_lengths) / max(1, len(token_lengths))),
        "mean_num_nodes": float(sum(node_counts) / max(1, len(node_counts))),
        "mean_num_edges": float(sum(edge_counts) / max(1, len(edge_counts))),
        "mean_depth": float(sum(depths) / max(1, len(depths))),
        "mean_time_span_minutes": float(sum(time_spans) / max(1, len(time_spans))),
        "url_rate": float(url_count / denom),
        "mention_rate": float(mention_count / denom),
        "hashtag_rate": float(hashtag_count / denom),
    }


def build_vocab_stats(source_texts: Dict[str, str], target_texts: Dict[str, str]) -> Dict[str, Any]:
    """Computes vocabulary overlap and salient token-frequency shifts."""
    source_tokens = Counter()
    target_tokens = Counter()
    for text in source_texts.values():
        source_tokens.update(tokenize(text))
    for text in target_texts.values():
        target_tokens.update(tokenize(text))
    source_vocab = set(source_tokens)
    target_vocab = set(target_tokens)
    overlap = source_vocab & target_vocab
    source_only = source_vocab - target_vocab
    target_only = target_vocab - source_vocab
    token_deltas = []
    for token in overlap:
        delta = float(target_tokens[token] - source_tokens[token])
        token_deltas.append((abs(delta), token, int(source_tokens[token]), int(target_tokens[token])))
    token_deltas.sort(reverse=True)
    return {
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "overlap_vocab_size": len(overlap),
        "source_oov_ratio_vs_target": float(len(source_only) / max(1, len(source_vocab))),
        "target_oov_ratio_vs_source": float(len(target_only) / max(1, len(target_vocab))),
        "top_frequency_deltas": [
            {
                "token": token,
                "source_count": src_count,
                "target_count": tgt_count,
            }
            for _, token, src_count, tgt_count in token_deltas[:20]
        ],
    }


def collect_embedding_stats(ds) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collects graph-level text embeddings and labels from one split."""
    embeddings: List[torch.Tensor] = []
    labels: List[int] = []
    for idx in range(len(ds)):
        sample = ds.get(idx)
        embeddings.append(sample.text_emb.view(-1).detach().cpu())
        labels.append(int(sample.y.view(-1)[0].item()))
    return torch.stack(embeddings, dim=0), torch.tensor(labels, dtype=torch.long)


def cosine_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Returns cosine distance for two vectors."""
    sim = torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
    return float(1.0 - sim)


def build_embedding_shift_report(source_ds, target_ds) -> Dict[str, Any]:
    """Builds domain-center and per-class embedding drift statistics."""
    source_emb, source_labels = collect_embedding_stats(source_ds)
    target_emb, target_labels = collect_embedding_stats(target_ds)
    source_center = source_emb.mean(dim=0)
    target_center = target_emb.mean(dim=0)
    report: Dict[str, Any] = {
        "domain_center_cosine_distance": cosine_distance(source_center, target_center),
        "domain_center_l2_distance": float(torch.norm(source_center - target_center, p=2).item()),
        "per_class_center_shift": {},
    }
    for class_idx in sorted(set(source_labels.tolist()) | set(target_labels.tolist())):
        source_mask = source_labels == int(class_idx)
        target_mask = target_labels == int(class_idx)
        if not source_mask.any() or not target_mask.any():
            continue
        source_proto = source_emb[source_mask].mean(dim=0)
        target_proto = target_emb[target_mask].mean(dim=0)
        report["per_class_center_shift"][str(int(class_idx))] = {
            "cosine_distance": cosine_distance(source_proto, target_proto),
            "l2_distance": float(torch.norm(source_proto - target_proto, p=2).item()),
            "source_support": int(source_mask.sum().item()),
            "target_support": int(target_mask.sum().item()),
        }
    return report


def write_markdown_report(path: str, report: Dict[str, Any]) -> None:
    """Writes a concise markdown view of the shift report."""
    lines = [
        f"# Transfer Shift Report: {report['source_dataset']} -> {report['target_dataset']}",
        "",
        "## Split Statistics",
    ]
    for domain_key in ("source", "target"):
        lines.append(f"### {domain_key.title()} ({report[f'{domain_key}_dataset']})")
        split_stats = report[f"{domain_key}_splits"]
        for split_name, stats in split_stats.items():
            lines.append(f"- `{split_name}`: n={stats['num_samples']}, labels={stats['label_counts']}, mean_tokens={stats['mean_text_tokens']:.2f}, mean_nodes={stats['mean_num_nodes']:.2f}, mean_edges={stats['mean_num_edges']:.2f}, mean_depth={stats['mean_depth']:.2f}, mean_span={stats['mean_time_span_minutes']:.2f}")
    vocab = report["vocab_shift"]
    lines.extend([
        "",
        "## Vocabulary Shift",
        f"- source_vocab={vocab['source_vocab_size']}, target_vocab={vocab['target_vocab_size']}, overlap={vocab['overlap_vocab_size']}",
        f"- source_oov_vs_target={vocab['source_oov_ratio_vs_target']:.4f}, target_oov_vs_source={vocab['target_oov_ratio_vs_source']:.4f}",
        "",
        "## Embedding Shift",
        f"- domain_center_cosine_distance={report['embedding_shift']['domain_center_cosine_distance']:.4f}",
        f"- domain_center_l2_distance={report['embedding_shift']['domain_center_l2_distance']:.4f}",
        "",
        "## Per-Class Drift",
    ])
    for class_idx, stats in report["embedding_shift"]["per_class_center_shift"].items():
        lines.append(f"- class={class_idx}: cosine_distance={stats['cosine_distance']:.4f}, l2_distance={stats['l2_distance']:.4f}, source_support={stats['source_support']}, target_support={stats['target_support']}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze transfer shift between source and target datasets.")
    parser.add_argument("--exp", required=True, help="Experiment config path.")
    parser.add_argument("--source", required=True, help="Source dataset name.")
    parser.add_argument("--target", required=True, help="Target dataset name.")
    parser.add_argument("--out_dir", default="", help="Optional output directory override.")
    parser.add_argument("--rebuild_cache", action="store_true", help="Force rebuild of text embedding caches.")
    args = parser.parse_args()

    exp_cfg = read_yaml_like(args.exp)
    dataset_yaml = read_yaml_like(os.path.join("Libs", "configs", "dataset.yaml")).get("dataset", {})
    raw_root = str(dataset_yaml.get("raw_root", os.path.join("Input", "raw")))
    processed_root = str(dataset_yaml.get("processed_root", os.path.join("Input", "processed")))
    text_runtime_cfg = resolve_dataset_text_runtime_cfg(exp_cfg)
    if args.rebuild_cache:
        text_runtime_cfg["rebuild_cache"] = True
    ensure_text_embedding_cache(args.source, text_runtime_cfg)
    ensure_text_embedding_cache(args.target, text_runtime_cfg)

    dataset_cfg = exp_cfg.get("dataset", {})
    prefix_cfg = dataset_cfg.get("prefix", {})
    observability_cfg = dataset_cfg.get("observability", {})
    use_lp_edges = bool(dataset_cfg.get("use_lp_edges", False))

    source_text_map_raw = load_text_mapping(args.source, raw_root)
    target_text_map_raw = load_text_mapping(args.target, raw_root)
    normalization_cfg = text_runtime_cfg.get("normalization", {})
    source_text_map = {sid: normalize_tweet_text(text, normalization_cfg) for sid, text in source_text_map_raw.items()}
    target_text_map = {sid: normalize_tweet_text(text, normalization_cfg) for sid, text in target_text_map_raw.items()}

    source_splits = {
        split: make_graph_dataset(
            root=os.path.join(processed_root, args.source),
            split=split,
            use_lp_edges=use_lp_edges,
            prefix_cfg=prefix_cfg,
            observability_cfg=observability_cfg,
            text_runtime_cfg=text_runtime_cfg,
        )
        for split in ("train", "val", "test")
    }
    target_splits = {
        split: make_graph_dataset(
            root=os.path.join(processed_root, args.target),
            split=split,
            use_lp_edges=use_lp_edges,
            prefix_cfg=prefix_cfg,
            observability_cfg=observability_cfg,
            text_runtime_cfg=text_runtime_cfg,
        )
        for split in ("train", "val", "test")
    }

    report = {
        "source_dataset": args.source,
        "target_dataset": args.target,
        "text_cache_name": text_runtime_cfg.get("cache_name", "text_embs.pt"),
        "source_splits": {split: summarize_split(ds, source_text_map_raw, source_text_map) for split, ds in source_splits.items()},
        "target_splits": {split: summarize_split(ds, target_text_map_raw, target_text_map) for split, ds in target_splits.items()},
        "vocab_shift": build_vocab_stats(source_text_map, target_text_map),
        "embedding_shift": build_embedding_shift_report(source_splits["train"], target_splits["train"]),
    }
    out_dir = args.out_dir or os.path.join(
        exp_cfg.get("outputs", {}).get("results_dir", os.path.join("Output", "results")),
        "transfer_shift",
    )
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"{args.source}_to_{args.target}_shift_report"
    json_path = os.path.join(out_dir, f"{base_name}.json")
    md_path = os.path.join(out_dir, f"{base_name}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_markdown_report(md_path, report)
    print(f"Saved shift report JSON to {json_path}")
    print(f"Saved shift report Markdown to {md_path}")


if __name__ == "__main__":
    main()
