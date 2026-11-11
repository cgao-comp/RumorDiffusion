"""Motivation experiment: demonstrating structural ambiguity in early-stage graphs.

This script implements the motivation experiment for the paper introduction,
showing that early-stage propagation graphs are structurally ambiguous while
mature-stage graphs are separable.
"""

import argparse
import csv
import logging
import os
import sys
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

# 抑制sklearn的UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Libs.data.dataset import GraphDataset  # noqa: E402
from Libs.models.baselines.GAE import create_gae_model  # noqa: E402
from Libs.utils.seed import set_global_seed  # noqa: E402


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        output_dir: Directory to save log file.
        
    Returns:
        Configured logger.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def append_rows_csv(csv_path: str, headers: list[str], rows: list[list[str]]) -> None:
    """Append rows to CSV, creating header if missing."""
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(headers)
        w.writerows(rows)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def recompute_structural_features(keep_indices: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  node_times: torch.Tensor) -> torch.Tensor:
    """Recompute structural features for a subgraph.
    
    Features computed: [normalized_time, depth, in_degree, out_degree]
    
    Args:
        keep_indices: Original indices of kept nodes.
        edge_index: Reindexed edge list of shape [2, num_edges].
        node_times: Node timestamps of kept nodes.
        
    Returns:
        Feature matrix of shape [num_nodes, 4].
    """
    num_nodes = len(keep_indices)
    
    # Normalized time
    if node_times.numel() > 1:
        min_time = node_times.min()
        max_time = node_times.max()
        if max_time > min_time:
            norm_time = (node_times - min_time).float() / (max_time - min_time).float()
        else:
            norm_time = torch.zeros(num_nodes, dtype=torch.float32)
    else:
        norm_time = torch.zeros(num_nodes, dtype=torch.float32)
    
    # Depth (simplified: use normalized time as proxy)
    depth = norm_time.clone()
    
    # Compute degrees
    in_degree = torch.zeros(num_nodes, dtype=torch.float32)
    out_degree = torch.zeros(num_nodes, dtype=torch.float32)
    
    if edge_index.numel() > 0:
        for i in range(num_nodes):
            in_degree[i] = (edge_index[1] == i).sum().float()
            out_degree[i] = (edge_index[0] == i).sum().float()
    
    # Stack features
    features = torch.stack([norm_time, depth, in_degree, out_degree], dim=1)
    return features


def create_early_subgraph_by_nodecount(data: Data, max_nodes: int = 15) -> Data:
    """Create early-stage subgraph by keeping first max_nodes nodes.
    
    Args:
        data: Original graph data.
        max_nodes: Maximum number of nodes to keep.
        
    Returns:
        Early-stage subgraph.
    """
    if data.num_nodes <= max_nodes:
        return data.clone()
    
    # Sort nodes by time
    sorted_indices = torch.argsort(data.node_time_minutes)
    keep_nodes = sorted_indices[:max_nodes]
    
    # Create node mapping
    node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(keep_nodes)}
    
    # Filter edges
    edge_mask = torch.isin(data.edge_index[0], keep_nodes) & \
                torch.isin(data.edge_index[1], keep_nodes)
    old_edges = data.edge_index[:, edge_mask]
    
    # Remap edges
    if old_edges.numel() > 0:
        new_edges = torch.tensor([
            [node_map[src.item()], node_map[dst.item()]]
            for src, dst in old_edges.t()
        ], dtype=torch.long).t().contiguous()
    else:
        new_edges = torch.empty((2, 0), dtype=torch.long)
    
    # Recompute features
    x_sub = recompute_structural_features(
        keep_nodes, new_edges, data.node_time_minutes[keep_nodes]
    )
    
    return Data(x=x_sub, edge_index=new_edges, y=data.y.clone())


def create_early_subgraph_by_timecutoff(data: Data, cutoff_minutes: int = 30) -> Data | None:
    """Create early-stage subgraph by time cutoff.
    
    Args:
        data: Original graph data.
        cutoff_minutes: Time cutoff in minutes.
        
    Returns:
        Early-stage subgraph.
    """
    # Filter nodes by time
    time_mask = data.node_time_minutes <= cutoff_minutes
    keep_indices = torch.where(time_mask)[0]
    
    if len(keep_indices) == 0:
        return None
    
    # Create node mapping
    node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(keep_indices)}
    
    # Filter edges
    edge_mask = torch.isin(data.edge_index[0], keep_indices) & \
                torch.isin(data.edge_index[1], keep_indices)
    old_edges = data.edge_index[:, edge_mask]
    
    # Remap edges
    if old_edges.numel() > 0:
        new_edges = torch.tensor([
            [node_map[src.item()], node_map[dst.item()]]
            for src, dst in old_edges.t()
        ], dtype=torch.long).t().contiguous()
    else:
        new_edges = torch.empty((2, 0), dtype=torch.long)
    
    # Recompute features
    x_sub = recompute_structural_features(
        keep_indices, new_edges, data.node_time_minutes[keep_indices]
    )
    
    return Data(x=x_sub, edge_index=new_edges, y=data.y.clone())


def load_dataset(dataset_name: str, data_root: str) -> List[Data]:
    """Load all data from all splits.
    
    Args:
        dataset_name: Name of dataset (e.g., 'twitter16').
        data_root: Root directory for processed data.
        
    Returns:
        List of all graph data objects.
    """
    dataset_path = os.path.join(data_root, dataset_name)
    all_graphs = []
    
    for split in ['train', 'val', 'test']:
        ds = GraphDataset(dataset_path, split=split, use_lp_edges=False, set_t_for_tgat=False)
        for i in range(len(ds)):
            data = ds.get(i)
            all_graphs.append(data)
    
    return all_graphs


def prepare_data_nodecount(all_graphs: List[Data], config: Dict) -> Tuple[List[Data], List[Data]]:
    """Prepare mature and early graphs using node count criterion.
    
    Args:
        all_graphs: All available graphs.
        config: Configuration for nodecount experiment.
        
    Returns:
        Tuple of (mature_graphs, early_graphs).
    """
    mature_candidates = config['mature_min_nodes'] if isinstance(config['mature_min_nodes'], list) else [config['mature_min_nodes']]
    early_candidates = config['early_max_nodes'] if isinstance(config['early_max_nodes'], list) else [config['early_max_nodes']]
    outputs = []
    for m_min in mature_candidates:
        for e_max in early_candidates:
            mature_graphs = []
            early_graphs = []
            for data in all_graphs:
                if data.num_nodes > m_min:
                    mature_graphs.append(data)
                early_sub = create_early_subgraph_by_nodecount(data, e_max)
                early_graphs.append(early_sub)
            outputs.append(((m_min, e_max), mature_graphs, early_graphs))
    return outputs


def prepare_data_timecutoff(all_graphs: List[Data], config: Dict) -> Tuple[List[Data], List[Data]]:
    """Prepare mature and early graphs using time cutoff criterion.
    
    Args:
        all_graphs: All available graphs.
        config: Configuration for timecutoff experiment.
        
    Returns:
        Tuple of (mature_graphs, early_graphs).
    """
    mature_candidates = config['mature_min_nodes'] if isinstance(config['mature_min_nodes'], list) else [config['mature_min_nodes']]
    cutoff_candidates = config['early_cutoff_minutes'] if isinstance(config['early_cutoff_minutes'], list) else [config['early_cutoff_minutes']]
    outputs = []
    for m_min in mature_candidates:
        for cutoff in cutoff_candidates:
            mature_graphs = []
            early_graphs = []
            for data in all_graphs:
                if data.num_nodes >= m_min:
                    mature_graphs.append(data)
                early_sub = create_early_subgraph_by_timecutoff(data, cutoff)
                if early_sub is not None:
                    early_graphs.append(early_sub)
            outputs.append(((m_min, cutoff), mature_graphs, early_graphs))
    return outputs


def filter_graphs_by_diversity(graphs: List[Data], min_edges: int = 5) -> List[Data]:
    """Filter graphs to ensure diversity and quality.
    
    Args:
        graphs: List of graph data objects.
        min_edges: Minimum number of edges required.
        
    Returns:
        Filtered list of graphs.
    """
    filtered_graphs = []
    for data in graphs:
        # 过滤条件：至少有一定数量的边和节点
        if data.edge_index.size(1) >= min_edges and data.num_nodes >= 3:
            filtered_graphs.append(data)
    
    return filtered_graphs


def ensure_class_diversity(graphs: List[Data], min_per_class: int = 2, logger: logging.Logger = None) -> List[Data]:
    """Ensure each class has minimum representation in the dataset.
    
    Args:
        graphs: List of graph data objects.
        min_per_class: Minimum number of graphs per class.
        logger: Logger instance for warnings.
        
    Returns:
        Filtered list of graphs with class diversity.
    """
    # 统计每个类别的数量
    class_counts = {}
    class_graphs = {}
    
    for data in graphs:
        label = data.y.item()
        if label not in class_counts:
            class_counts[label] = 0
            class_graphs[label] = []
        class_counts[label] += 1
        class_graphs[label].append(data)
    
    # 过滤掉数量不足的类别
    filtered_graphs = []
    for label, count in class_counts.items():
        if count >= min_per_class:
            filtered_graphs.extend(class_graphs[label])
        else:
            if logger:
                logger.warning(f"Class {label} has only {count} graphs, skipping")
    
    return filtered_graphs


def train_gae(graphs: List[Data], config: Dict, logger: logging.Logger) -> torch.nn.Module:
    """Train Graph Autoencoder on mature graphs.
    
    Args:
        graphs: List of training graphs.
        config: GAE training configuration.
        logger: Logger instance.
        
    Returns:
        Trained GAE model.
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = create_gae_model(
        in_channels=config['in_channels'],
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # 数据过滤和多样性保证
    logger.info(f"Original graphs: {len(graphs)}")
    graphs = filter_graphs_by_diversity(graphs, min_edges=5)
    logger.info(f"After edge filtering: {len(graphs)}")
    graphs = ensure_class_diversity(graphs, min_per_class=2, logger=logger)
    logger.info(f"After class diversity filtering: {len(graphs)}")
    
    if len(graphs) < 10:
        raise ValueError(f"Not enough graphs after filtering: {len(graphs)}")
    
    # 按类别分层分割，确保训练和验证集都有所有类别
    class_graphs = {}
    for data in graphs:
        label = data.y.item()
        if label not in class_graphs:
            class_graphs[label] = []
        class_graphs[label].append(data)
    
    train_graphs = []
    val_graphs = []
    
    for label, class_data in class_graphs.items():
        # 随机打乱每个类别的数据
        np.random.shuffle(class_data)
        # 按比例分割
        split_idx = int(len(class_data) * config['train_ratio'])
        train_graphs.extend(class_data[:split_idx])
        val_graphs.extend(class_data[split_idx:])
    
    # 再次打乱
    np.random.shuffle(train_graphs)
    np.random.shuffle(val_graphs)
    
    logger.info(f"Training on {len(train_graphs)} graphs, validating on {len(val_graphs)} graphs")
    logger.info(f"Device: {device}")
    
    # 统计类别分布
    train_labels = [g.y.item() for g in train_graphs]
    val_labels = [g.y.item() for g in val_graphs]
    logger.info(f"Train class distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    logger.info(f"Val class distribution: {dict(zip(*np.unique(val_labels, return_counts=True)))}")
    
    best_val_auc = 0.0
    best_model_state = None
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        
        for data in tqdm(train_graphs, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Encode
            z = model.encode(data.x, data.edge_index)
            
            # Negative sampling
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.edge_index.size(1)
            )
            
            # Compute loss
            loss = model.recon_loss(z, data.edge_index, neg_edge_index)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_graphs)
        
        # Validation - 更鲁棒的验证逻辑
        model.eval()
        val_aucs = []
        val_losses = []
        
        with torch.no_grad():
            for data in val_graphs:
                data = data.to(device)
                z = model.encode(data.x, data.edge_index)
                
                # 生成负样本边进行验证
                try:
                    neg_edge_index = negative_sampling(
                        edge_index=data.edge_index,
                        num_nodes=data.num_nodes,
                        num_neg_samples=data.edge_index.size(1)
                    )
                    
                    # 计算重构损失作为验证指标
                    loss = model.recon_loss(z, data.edge_index, neg_edge_index)
                    val_losses.append(loss.item())
                    
                    # 尝试计算AUC，如果失败则跳过
                    try:
                        auc = model.test(z, data.edge_index, neg_edge_index)
                        if not np.isnan(auc) and auc > 0:
                            val_aucs.append(auc)
                    except (ValueError, RuntimeError, ZeroDivisionError):
                        # 跳过无法计算AUC的图
                        continue
                        
                except (ValueError, RuntimeError):
                    # 跳过无法生成负样本的图
                    continue
        
        # 使用平均损失作为主要验证指标，AUC作为辅助指标
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_val_auc = np.mean(val_aucs) if val_aucs else 0.0
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val AUC={avg_val_auc:.4f}")
        
        # Save best model based on validation loss (lower is better)
        if avg_val_loss < best_val_auc:  # 重用变量名，但现在是损失值
            best_val_auc = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    logger.info(f"Training complete. Best Val AUC: {best_val_auc:.4f}")
    return model


def generate_embeddings(model: torch.nn.Module,
                       graphs: List[Data],
                       config: Dict,
                       logger: logging.Logger = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate graph-level embeddings using mean pooling.
    
    Args:
        model: Trained GAE model.
        graphs: List of graphs to encode.
        config: Configuration dictionary.
        logger: Logger instance for warnings.
        
    Returns:
        Tuple of (embeddings, labels).
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    
    embeddings = []
    labels = []
    failed_count = 0
    
    with torch.no_grad():
        for data in tqdm(graphs, desc="Generating embeddings"):
            try:
                data = data.to(device)
                z = model.encode(data.x, data.edge_index)
                graph_emb = z.mean(dim=0)  # Mean pooling
                embeddings.append(graph_emb.cpu())
                labels.append(data.y.cpu())
            except Exception:
                # 跳过有问题的图
                failed_count += 1
                continue
    
    if failed_count > 0:
        if logger:
            logger.warning(f"Failed to generate embeddings for {failed_count} graphs")
    
    if len(embeddings) == 0:
        raise ValueError("No valid embeddings generated")
    
    embeddings = torch.stack(embeddings)
    labels = torch.cat(labels)
    
    return embeddings, labels


def visualize_embeddings(embeddings: torch.Tensor,
                        labels: torch.Tensor,
                        output_path: str,
                        title: str,
                        config: Dict,
                        logger: logging.Logger) -> None:
    """Generate and save t-SNE visualization.
    
    Args:
        embeddings: Graph embeddings of shape [num_graphs, embedding_dim].
        labels: Graph labels of shape [num_graphs].
        output_path: Path to save the visualization.
        title: Title for the plot.
        config: Visualization configuration.
        logger: Logger instance.
    """
    logger.info(f"Running t-SNE for {len(embeddings)} graphs...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=config['perplexity'],
        max_iter=config['max_iter'],
        random_state=config['random_state']
    )
    emb_2d = tsne.fit_transform(embeddings.numpy())
    
    fig, ax = plt.subplots(figsize=tuple(config['figsize']))
    
    unique_labels = sorted(set(labels.numpy()))
    for label_id in unique_labels:
        mask = labels == label_id
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=config['colors'].get(label_id, '#999999'),
            marker=config['markers'].get(label_id, 'o'),
            alpha=config['alpha'],
            label=config['legend_labels'].get(label_id, f'Class {label_id}'),
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {output_path}")


def evaluate_embeddings(embeddings: torch.Tensor,
                        labels: torch.Tensor,
                        lin_cfg: Dict,
                        seed: int) -> Dict[str, float]:
    """Compute clusterability metrics and linear probe macro F1."""
    X = embeddings.cpu().numpy()
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return {"silhouette": 0.0, "davies_bouldin": 0.0, "calinski_harabasz": 0.0, "linear_macro_f1": 0.0}
    metrics = {
        "silhouette": float(silhouette_score(X, y)) if len(np.unique(y)) > 1 else 0.0,
        "davies_bouldin": float(davies_bouldin_score(X, y)) if len(np.unique(y)) > 1 else 0.0,
        "calinski_harabasz": float(calinski_harabasz_score(X, y)) if len(np.unique(y)) > 1 else 0.0,
    }
    test_size = 1.0 - float(lin_cfg.get("train_ratio", 0.8))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = LogisticRegression(
        max_iter=int(lin_cfg.get("max_iter", 200)),
        C=float(lin_cfg.get("C", 1.0)),
        multi_class="auto",
        solver="lbfgs",
        n_jobs=None,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics["linear_macro_f1"] = float(f1_score(y_test, y_pred, average="macro"))
    return metrics


def run_experiment(experiment_type: str,
                  all_graphs: List[Data],
                  config: Dict,
                  logger: logging.Logger,
                  seed: int,
                  metrics_rows: List[List[str]],
                  lin_cfg: Dict) -> None:
    """Run a single experiment (nodecount or timecutoff) over all config combos."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {experiment_type.upper()} experiment (seed={seed})")
    logger.info(f"{'='*60}\n")
    combos = prepare_data_nodecount(all_graphs, config[experiment_type]) if experiment_type == 'nodecount' else prepare_data_timecutoff(all_graphs, config[experiment_type])
    for (m_param, e_param), mature_graphs, early_graphs in combos:
        if len(mature_graphs) == 0 or len(early_graphs) == 0:
            logger.warning(f"Skip combo m={m_param}, e={e_param} due to insufficient graphs")
            continue
        if len(mature_graphs) < 10 or len(early_graphs) < 10:
            logger.warning(f"Skip combo m={m_param}, e={e_param} due to too few graphs (<10)")
            continue
        logger.info(f"[{experiment_type}] combo m={m_param}, e={e_param} -> mature {len(mature_graphs)}, early {len(early_graphs)}")
        mature_nodes = [g.num_nodes for g in mature_graphs]
        early_nodes = [g.num_nodes for g in early_graphs]
        logger.info(f"Mature nodes mean/std: {np.mean(mature_nodes):.1f}/{np.std(mature_nodes):.1f}; Early nodes mean/std: {np.mean(early_nodes):.1f}/{np.std(early_nodes):.1f}")
        mature_labels = [g.y.item() for g in mature_graphs]
        early_labels = [g.y.item() for g in early_graphs]
        logger.info(f"Mature label distribution: {dict(zip(*np.unique(mature_labels, return_counts=True)))}")
        logger.info(f"Early label distribution: {dict(zip(*np.unique(early_labels, return_counts=True)))}")
        model = train_gae(mature_graphs, config['gae'], logger)
        tag = f"{experiment_type}_m{m_param}_e{e_param}_seed{seed}"
        model_path = f"{config['experiment']['output_dir']}/gae_{tag}.pt"
        torch.save(model.state_dict(), model_path)
        mature_embs, mature_labels_tensor = generate_embeddings(model, mature_graphs, config['gae'], logger)
        early_embs, early_labels_tensor = generate_embeddings(model, early_graphs, config['gae'], logger)
        emb_path = f"{config['experiment']['output_dir']}/embeddings_{tag}.pt"
        torch.save({
            'mature_embeddings': mature_embs,
            'mature_labels': mature_labels_tensor,
            'early_embeddings': early_embs,
            'early_labels': early_labels_tensor
        }, emb_path)
        mature_metrics = evaluate_embeddings(mature_embs, mature_labels_tensor, lin_cfg, seed)
        early_metrics = evaluate_embeddings(early_embs, early_labels_tensor, lin_cfg, seed)
        metrics_rows.append([
            config['experiment']['current_dataset'],
            experiment_type,
            str(m_param),
            str(e_param),
            str(seed),
            "mature",
            f"{mature_metrics['silhouette']:.6f}",
            f"{mature_metrics['davies_bouldin']:.6f}",
            f"{mature_metrics['calinski_harabasz']:.6f}",
            f"{mature_metrics['linear_macro_f1']:.6f}",
        ])
        metrics_rows.append([
            config['experiment']['current_dataset'],
            experiment_type,
            str(m_param),
            str(e_param),
            str(seed),
            "early",
            f"{early_metrics['silhouette']:.6f}",
            f"{early_metrics['davies_bouldin']:.6f}",
            f"{early_metrics['calinski_harabasz']:.6f}",
            f"{early_metrics['linear_macro_f1']:.6f}",
        ])
    visualize_embeddings(
        mature_embs, mature_labels_tensor,
            f"{config['experiment']['output_dir']}/tsne_{tag}_mature.png",
        f"Mature Stage ({experiment_type.title()})",
        config['visualization'],
        logger
    )
    visualize_embeddings(
        early_embs, early_labels_tensor,
            f"{config['experiment']['output_dir']}/tsne_{tag}_early.png",
        f"Early Stage ({experiment_type.title()})",
        config['visualization'],
        logger
    )


def main():
    """Main experiment entry point."""
    parser = argparse.ArgumentParser(description="GAE Motivation Experiment")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--experiment', type=str, choices=['nodecount', 'timecutoff', 'both'],
                       default=None, help="Which experiment to run (overrides config)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    seeds = config['experiment'].get('seeds', [config['experiment'].get('seed', 42)])
    datasets = config['experiment'].get('datasets', [config['experiment'].get('dataset')])
    run_experiments = args.experiment if args.experiment else config['experiment'].get('run_experiments', 'both')
    data_root = config['experiment'].get('data_root', 'Input/processed')
    lin_cfg = config.get('gae', {}).get('linear_probe', {})
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    metrics_path = os.path.join(config['experiment']['output_dir'], "metrics.csv")
    metrics_headers = ["dataset", "experiment", "mature_param", "early_param", "seed", "stage", "silhouette", "davies_bouldin", "calinski_harabasz", "linear_macro_f1"]
    logger = setup_logging(config['experiment']['output_dir'])
    logger.info("="*60)
    logger.info("GAE Motivation Experiment")
    logger.info("="*60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output dir: {config['experiment']['output_dir']}")
    metrics_rows: List[List[str]] = []
    for ds in datasets:
        config['experiment']['current_dataset'] = ds
        logger.info(f"\nLoading dataset: {ds}")
        all_graphs = load_dataset(ds, data_root)
        logger.info(f"Loaded {len(all_graphs)} graphs")
        for seed in seeds:
            set_global_seed(int(seed))
            if run_experiments in ['nodecount', 'both']:
                run_experiment('nodecount', all_graphs, config, logger, seed, metrics_rows, lin_cfg)
            if run_experiments in ['timecutoff', 'both']:
                run_experiment('timecutoff', all_graphs, config, logger, seed, metrics_rows, lin_cfg)
    append_rows_csv(metrics_path, metrics_headers, metrics_rows)
    logger.info("\nAll experiments completed successfully!")


if __name__ == '__main__':
    main()
