"""Dataset analysis script for RumorDiffusion.

This script analyzes the distribution and characteristics of different classes
in the processed dataset to help diagnose classification issues.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List

import torch
from torch_geometric.data import Data


def load_dataset(dataset_name: str, processed_root: str = "Input/processed"):
    """Loads processed PyG dataset and splits.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'twitter15').
        processed_root: Root directory of processed datasets.
    
    Returns:
        Tuple of (data_list, splits_dict).
    """
    dataset_dir = os.path.join(processed_root, dataset_name, "pyg")
    data_path = os.path.join(dataset_dir, "data.pt")
    slices_path = os.path.join(dataset_dir, "slices.pt")
    splits_path = os.path.join(processed_root, dataset_name, "splits.json")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data_obj = torch.load(data_path)
    slices = torch.load(slices_path)
    
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    # Reconstruct individual Data objects
    data_list: List[Data] = []
    num_samples = len(slices['x']) - 1
    
    for i in range(num_samples):
        data = Data()
        for key in slices.keys():
            if key == 'y':
                s = slices[key]
                data.y = data_obj.y[s[i]:s[i+1]]
            elif key == 'x':
                s = slices[key]
                data.x = data_obj.x[s[i]:s[i+1]]
            elif key == 'edge_index':
                s = slices[key]
                data.edge_index = data_obj.edge_index[:, s[i]:s[i+1]]
            elif key == 'root_index':
                s = slices[key]
                data.root_index = data_obj.root_index[s[i]:s[i+1]]
        data_list.append(data)
    
    return data_list, splits


def analyze_class_distribution(data_list: List[Data], splits: Dict, class_names: Dict[int, str]):
    """Analyzes class distribution across splits.
    
    Args:
        data_list: List of Data objects.
        splits: Dictionary mapping split names to indices.
        class_names: Mapping from class id to class name.
    """
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for split_name in ['train', 'val', 'test']:
        indices = splits[split_name]
        class_counts = defaultdict(int)
        
        for idx in indices:
            label = int(data_list[idx].y.item())
            class_counts[label] += 1
        
        total = len(indices)
        print(f"\n{split_name.upper()} SET ({total} samples):")
        print("-" * 60)
        
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / total) * 100
            class_name = class_names.get(class_id, f"class_{class_id}")
            print(f"  {class_name:12s}: {count:4d} samples ({percentage:5.2f}%)")


def analyze_graph_statistics(data_list: List[Data], splits: Dict, class_names: Dict[int, str]):
    """Analyzes graph-level statistics per class.
    
    Args:
        data_list: List of Data objects.
        splits: Dictionary mapping split names to indices.
        class_names: Mapping from class id to class name.
    """
    print("\n" + "="*60)
    print("GRAPH STATISTICS BY CLASS (Training Set)")
    print("="*60)
    
    train_indices = splits['train']
    class_stats = defaultdict(lambda: {'num_nodes': [], 'num_edges': [], 'avg_degree': []})
    
    for idx in train_indices:
        data = data_list[idx]
        label = int(data.y.item())
        num_nodes = int(data.x.size(0))
        num_edges = int(data.edge_index.size(1))
        avg_degree = num_edges / max(1, num_nodes)
        
        class_stats[label]['num_nodes'].append(num_nodes)
        class_stats[label]['num_edges'].append(num_edges)
        class_stats[label]['avg_degree'].append(avg_degree)
    
    print(f"\n{'Class':<12} {'Count':<7} {'Avg Nodes':<12} {'Avg Edges':<12} {'Avg Degree':<12}")
    print("-" * 60)
    
    for class_id in sorted(class_stats.keys()):
        stats = class_stats[class_id]
        class_name = class_names.get(class_id, f"class_{class_id}")
        count = len(stats['num_nodes'])
        avg_nodes = sum(stats['num_nodes']) / count
        avg_edges = sum(stats['num_edges']) / count
        avg_degree = sum(stats['avg_degree']) / count
        
        print(f"{class_name:<12} {count:<7} {avg_nodes:<12.2f} {avg_edges:<12.2f} {avg_degree:<12.2f}")
    
    # Detailed statistics for each class
    print("\n" + "="*60)
    print("DETAILED GRAPH SIZE DISTRIBUTION")
    print("="*60)
    
    for class_id in sorted(class_stats.keys()):
        stats = class_stats[class_id]
        class_name = class_names.get(class_id, f"class_{class_id}")
        nodes = sorted(stats['num_nodes'])
        count = len(nodes)
        
        print(f"\n{class_name.upper()} (n={count}):")
        print(f"  Nodes: min={min(nodes)}, max={max(nodes)}, median={nodes[count//2]}")
        print(f"  Percentiles: 25th={nodes[count//4]}, 75th={nodes[3*count//4]}, 95th={nodes[int(0.95*count)]}")


def main():
    """Main analysis function."""
    dataset_name = "twitter15"
    processed_root = "Input/processed"
    
    class_names = {
        0: "non-rumor",
        1: "true",
        2: "false",
        3: "unverified"
    }
    
    print(f"\nLoading dataset: {dataset_name}")
    data_list, splits = load_dataset(dataset_name, processed_root)
    print(f"Total samples: {len(data_list)}")
    
    analyze_class_distribution(data_list, splits, class_names)
    analyze_graph_statistics(data_list, splits, class_names)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

