import argparse
import csv
import json
import os
from typing import List, Dict


def load_records(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def select_top(records: List[Dict], key: str, descending: bool, topk: int) -> List[Dict]:
    sorted_rec = sorted(records, key=lambda r: float(r.get(key, 0.0)), reverse=descending)
    return sorted_rec[: topk]


def save_csv(path: str, rows: List[List[str]]) -> None:
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = ["type", "source_id", "pred_class", "label", "pred_prob", "pair_margin", "cls_norm", "token_norm_mean", "token_norm_max"]
    exists = os.path.isfile(path)
    with open(path, "a", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(headers)
        w.writerows(rows)


def summarize(input_path: str, out_path: str, topk: int) -> None:
    records = load_records(input_path)
    if len(records) == 0:
        return
    for r in records:
        r.setdefault("label", -1)
        r.setdefault("correct", 0)
    correct = [r for r in records if int(r.get("correct", 0)) == 1]
    wrong = [r for r in records if int(r.get("correct", 0)) == 0]
    uncertain = sorted(records, key=lambda r: abs(float(r.get("pair_margin", 0.0))))
    rows: List[List[str]] = []
    for rec_type, subset in [
        ("confident_correct", select_top(correct, "pred_prob", True, topk)),
        ("confident_wrong", select_top(wrong, "pred_prob", True, topk)),
        ("uncertain", uncertain[:topk]),
    ]:
        for r in subset:
            rows.append([
                rec_type,
                r.get("source_id", ""),
                str(r.get("pred_class", "")),
                str(r.get("label", "")),
                f"{float(r.get('pred_prob', 0.0)):.6f}",
                f"{float(r.get('pair_margin', 0.0)):.6f}",
                f"{float(r.get('cls_norm', 0.0)):.6f}",
                f"{float(r.get('token_norm_mean', 0.0)):.6f}",
                f"{float(r.get('token_norm_max', 0.0)):.6f}",
            ])
    save_csv(out_path, rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize explain outputs into CSV selections.")
    parser.add_argument("--input", required=True, help="Path to explain jsonl file")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--topk", type=int, default=20, help="Top K samples per category")
    args = parser.parse_args()
    summarize(args.input, args.out, args.topk)


if __name__ == "__main__":
    main()

