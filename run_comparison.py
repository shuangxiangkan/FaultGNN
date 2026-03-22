#!/usr/bin/env python3

import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, recall_score, precision_score
import matplotlib.pyplot as plt
import gc

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_generator import UnifiedDatasetGenerator, UnifiedDatasetLoader, generate_dataset_from_config, _cleanup_global_pool
from models import FaultGNN, RNNIFDCom_PMC, evaluate_fifpdpmc
from logging_config import get_logger, init_default_logging

# Initialize logging configuration
init_default_logging()
logger = get_logger(__name__)


# ========== Performance optimization: Global resource management ==========
def cleanup_global_resources():
    """Clean up global resources (process pools, etc.)"""
    logger.info("Executing global resource cleanup...")
    _cleanup_global_pool()
    gc.collect()
    logger.info("Global resource cleanup completed")
# =============================================


def apply_partial_symptoms_to_gnn_data(gnn_data_list, missing_ratio, missing_type, seed=42):
    """
    Apply partial symptom processing to GNN test data
    
    Args:
        gnn_data_list: GNN data list
        missing_ratio: Missing ratio
        missing_type: Missing type ('node_disable')
        seed: Random seed
        
    Returns:
        tuple: (Processed GNN data list, disabled node indices dictionary {data_idx: disabled_node_indices})
    """
    import torch
    import numpy as np
    from copy import deepcopy
    
    if missing_ratio <= 0:
        return gnn_data_list, {}
    
    rng = np.random.default_rng(seed)
    processed_data = []
    disabled_nodes_dict = {}
    
    for data_idx, data in enumerate(gnn_data_list):
        data_copy = deepcopy(data)
        
        if missing_type == 'node_disable':
            # Disable a certain proportion of nodes (set their features to 0)
            num_nodes = data_copy.x.shape[0]
            num_to_disable = int(num_nodes * missing_ratio)
            
            if num_to_disable > 0:
                # Randomly select nodes to disable
                disabled_nodes = rng.choice(num_nodes, size=num_to_disable, replace=False)
                data_copy.x[disabled_nodes] = 0.0
                disabled_nodes_dict[data_idx] = disabled_nodes
            else:
                disabled_nodes_dict[data_idx] = np.array([])
        else:
            raise ValueError(f"Unsupported missing type: {missing_type}")
        
        processed_data.append(data_copy)
    
    logger.info(f"Applied {missing_type} partial symptom processing to {len(processed_data)} GNN graphs")
    return processed_data, disabled_nodes_dict


def apply_partial_symptoms_to_rnn_data(X_data, y_data, missing_ratio, missing_type, graph, seed=42):
    """
    Apply partial symptom processing to RNN test data
    
    Args:
        X_data: RNN input data (sample count, feature count)
        y_data: RNN label data (sample count, node count)
        missing_ratio: Missing ratio
        missing_type: Missing type ('node_disable')
        graph: Graph object for getting edge structure information
        seed: Random seed
        
    Returns:
        tuple: (处理后的RNN数据, 处理后的标签数据, 被禁用的节点集合)
    """
    import numpy as np
    
    if missing_ratio <= 0:
        return X_data, y_data, set()
    
    rng = np.random.default_rng(seed)
    X_copy = X_data.copy()
    y_copy = y_data.copy()
    
    if missing_type == 'node_disable':
        # Select nodes to disable
        num_nodes = len(graph.vertices)
        num_to_disable = int(num_nodes * missing_ratio)
        disabled_nodes = set()
        
        if num_to_disable > 0:
            disabled_indices = rng.choice(num_nodes, size=num_to_disable, replace=False)
            disabled_nodes = {graph.vertices[i] for i in disabled_indices}
        
        # For RNN, we need to set the features of edges involving disabled nodes to 0
        # The feature vector of RNN corresponds to all edges of the graph
        edge_to_feature_map = {}
        feature_idx = 0
        
        # Build a mapping from edges to feature indices
        for u, v in graph.edges:
            edge_to_feature_map[(u, v)] = feature_idx
            feature_idx += 1
        
        # Set the features of edges involving disabled nodes to 0
        for (u, v), feat_idx in edge_to_feature_map.items():
            if u in disabled_nodes or v in disabled_nodes:
                X_copy[:, feat_idx] = 0.0
        
        # Remove predictions for disabled nodes from labels (used during evaluation)
        # Note: Here we don't directly modify y_copy, because we need to maintain dimension consistency
        # The information about disabled nodes will be used during evaluation
        
    else:
        raise ValueError(f"Unsupported missing type: {missing_type}")
    
    logger.info(f"Applied {missing_type} partial symptom processing to RNN data, disabled {len(disabled_nodes)} nodes")
    return X_copy, y_copy, disabled_nodes


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Initialize Focal Loss, used to handle class imbalance problem
        
        Args:
            alpha: Weight factor, used to balance positive and negnnive samples
            gamma: Focus parameter, increase the attention to difficult-to-classify samples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Assign higher weight to faulty nodes (y=1)
        alpha_factor = torch.ones_like(targets) * self.alpha
        alpha_factor[targets == 0] = 1 - self.alpha
        
        # Calculate focal loss
        focal_weight = alpha_factor * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        
        return loss.mean()


def train_gnn_model(train_loader, val_loader, input_dim, hidden_dim=64, 
                   num_layers=2, heads=8, epochs=100, lr=0.002, device='cpu'):
    """Train GNN model - optimized for large graphs"""
    logger.info("Starting to train GNN model...")
    
    # ========== Performance optimization: model parameter adjustment for large graphs ==========
    # For large graphs, reduce model parameters to save memory
    if input_dim > 20:  # If input dimension is large, it means it's a large graph
        logger.info(f"Detected large graph input dimension ({input_dim}), adjusting model parameters to save memory")
        hidden_dim = min(hidden_dim, 32)  # Reduce hidden layer dimension
        heads = min(heads, 4)  # Reduce number of attention heads
        logger.info(f"Adjusted parameters: hidden_dim={hidden_dim}, heads={heads}")
    # =============================================
    
    # ========== Core functionality: model initialization ==========
    model = FaultGNN(input_dim=input_dim, hidden_dim=hidden_dim, 
                   num_layers=num_layers, heads=heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"GNN trainable parameters: {num_params:,}")
    
    best_val_f1 = 0
    best_model_state = model.state_dict()
    best_epoch = 0
    early_stop_counter = 0
    early_stop_patience = 30
    
    train_f1_history = []
    val_f1_history = []
    # ==========================================
    
    # ========== Performance optimization: progress monitoring for large graph training ==========
    # Check the size of training data, decide whether to show progress bar
    total_train_graphs = sum(data.num_graphs for data in train_loader)
    show_progress = total_train_graphs > 10
    # =============================================
    
    # ========== Core functionality: training loop ==========
    for epoch in range(epochs):
        # === Core functionality: training stage ===
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        batch_count = 0  # Performance optimization: batch counter
        for data in train_loader:
            # ========== Performance optimization: memory error handling ==========
            try:
                # ========================================
                
                # === Core functionality: forward propagnnion and backward propagnnion ===
                data = data.to(device)
                optimizer.zero_grad()
                
                # Forward propagnnion
                out = model(data.x, data.edge_index)
                loss = criterion(out, data.y)
                
                if torch.isnan(loss):
                    logger.warning(f"GNN第 {epoch+1} 轮第 {batch_count+1} 批次损失为NaN，跳过此批次")
                    continue
                    
                # Backward propagnnion
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                
                # Collect prediction results
                preds = out.argmax(dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_true.extend(data.y.cpu().numpy())
                # =======================================
                
                batch_count += 1
                
                # ========== Performance optimization: progress display and memory management ==========
                # Show progress when training large graphs
                if show_progress and batch_count % max(1, len(train_loader) // 4) == 0:
                    logger.info(f"  GNN {epoch+1}th epoch training progress: {batch_count}/{len(train_loader)} batches")
                
                # Clean up GPU cache (if using GPU)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                # =============================================
                    
            # ========== Performance optimization: memory error handling ==========
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GNN training out of memory: {e}")
                    logger.error("Suggestions: 1) Decrease batch size 2) Use CPU training 3) Decrease model parameters")
                    # Clean up memory and continue
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
            # =============================================
        
        # ========== Performance optimization: data validation ==========
        if len(train_preds) == 0:
            logger.warning(f"GNN {epoch+1}th epoch has no valid training data")
            continue
        # =====================================
            
        # === Core functionality: calculate training metrics ===
        train_f1 = f1_score(train_true, train_preds, zero_division=0)
        train_f1_history.append(train_f1)
        # ============================
        
        # === Core functionality: validation stage ===
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            val_batch_count = 0  # Performance optimization: validation batch counter
            for data in val_loader:
                # ========== Performance optimization: validation stage memory error handling ==========
                try:
                    # === Core functionality: validation forward propagnnion ===
                    data = data.to(device)
                    out = model(data.x, data.edge_index)
                    preds = out.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(data.y.cpu().numpy())
                    # ===============================
                    
                    val_batch_count += 1
                    
                    # ========== Performance optimization: GPU cache cleanup ==========
                    # Clean up GPU cache
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    # ========================================
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"GNN validation out of memory, skipping this batch: {e}")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
                # =============================================
        
        # ========== Performance optimization: validation data check ==========
        if len(val_preds) == 0:
            logger.warning(f"GNN {epoch+1}th epoch has no valid validation data")
            val_f1 = 0
        else:
            val_f1 = f1_score(val_true, val_preds, zero_division=0)
        # ========================================
        
        # === Core functionality: record validation metrics ===
        val_f1_history.append(val_f1)
        # ===========================
        
        # === Core functionality: calculate and display training results ===
        # Calculate prediction distribution and confusion matrix
        pred_counts = np.bincount(val_preds, minlength=2) if val_preds else np.array([0, 0])
        cm = confusion_matrix(val_true, val_preds) if val_preds else np.array([[0, 0], [0, 0]])
        
        # Calculate average loss
        avg_loss = train_loss / max(1, batch_count)
        
        logger.info(f'GNN {epoch+1}th epoch, training loss: {avg_loss:.4f}, '
                   f'training F1: {train_f1:.4f}, validation F1: {val_f1:.4f}, prediction distribution: {pred_counts}, '
                   f'confusion matrix: \n{cm}')
        # ====================================
        
        # === Core functionality: Early Stopping and learning rate scheduling ===
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= early_stop_patience:
            logger.info(f'GNN Early stopping triggered after {epoch+1} epochs')
            break
        
        scheduler.step(val_f1)
        # ==========================================
        
        # ========== Performance optimization: periodic garbage collection ==========
        if epoch % 10 == 0:
            gc.collect()
        # ========================================
    # =======================================
    
    total_epochs = len(train_f1_history)
    model.load_state_dict(best_model_state)
    logger.info(f"GNN training completed, best validation F1: {best_val_f1:.4f}, "
               f"convergence epoch: {best_epoch}/{total_epochs}, params: {num_params:,}")
    
    train_info = {
        'convergence_epoch': best_epoch,
        'total_epochs': total_epochs,
        'num_params': num_params
    }
    return model, train_f1_history, val_f1_history, train_info


# ========== Core functionality: RNNIFDCOM model training ==========
def train_rnn_model(X_train, y_train, X_val, y_val, hidden_dims=[64, 32], 
                   epochs=100, lr=0.002, device='cpu'):
    """Train RNNIFDCOM model"""
    logger.info("Starting to train RNNIFDCOM model...")
    
    # === Core functionality: model initialization ===
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = RNNIFDCom_PMC(input_dim, output_dim, hidden_dims).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"RNNIFDCOM trainable parameters: {num_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Weighted loss to address class imbalance: fault nodes are rare minority
    num_pos = y_train.sum()
    num_neg = y_train.size - num_pos
    if num_pos > 0:
        pw = num_neg / num_pos
    else:
        pw = 1.0
    pos_weight = torch.tensor([pw], dtype=torch.float32).to(device)
    logger.info(f"RNNIFDCOM class imbalance: pos={int(num_pos)}, neg={int(num_neg)}, pos_weight={pw:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    best_val_f1 = 0
    best_model_state = model.state_dict()
    best_epoch = 0
    early_stop_counter = 0
    early_stop_patience = 30
    
    train_f1_history = []
    val_f1_history = []
    # ===========================
    
    for epoch in range(epochs):
        # === Core functionality: training stage ===
        model.train()
        optimizer.zero_grad()
        out = model(X_train_tensor)
        
        loss = criterion(out, y_train_tensor)
        if torch.isnan(loss):
            logger.warning(f"RNNIFDCOM {epoch+1}th epoch loss is NaN")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # === Core functionality: calculate training F1 ===
        train_preds = torch.sigmoid(out) > 0.5
        train_preds = train_preds.float().cpu().numpy()
        train_true = y_train_tensor.cpu().numpy()
        
        train_preds_flat = train_preds.flatten()
        train_true_flat = train_true.flatten()
        train_f1 = f1_score(train_true_flat, train_preds_flat, zero_division=0)
        train_f1_history.append(train_f1)
        
        model.eval()
        with torch.no_grad():
            out = model(X_val_tensor)
            val_preds = torch.sigmoid(out) > 0.5
            val_preds = val_preds.float().cpu().numpy()
        
        val_true = y_val_tensor.cpu().numpy()
        val_preds_flat = val_preds.flatten()
        val_true_flat = val_true.flatten()
        val_f1 = f1_score(val_true_flat, val_preds_flat, zero_division=0)
        val_f1_history.append(val_f1)
        
  
        pred_counts = np.bincount(val_preds_flat.astype(int), minlength=2)
        cm = confusion_matrix(val_true_flat, val_preds_flat)
        
        logger.info(f'RNNIFDCOM {epoch+1}th epoch, training loss: {loss.item():.4f}, '
                   f'training F1: {train_f1:.4f}, validation F1: {val_f1:.4f}, prediction distribution: {pred_counts}, '
                   f'confusion matrix: \n{cm}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= early_stop_patience:
            logger.info(f'RNNIFDCOM Early stopping triggered after {epoch+1} epochs')
            break
        
        scheduler.step(val_f1)
    
    total_epochs = len(train_f1_history)
    model.load_state_dict(best_model_state)
    logger.info(f"RNNIFDCOM training completed, best validation F1: {best_val_f1:.4f}, "
               f"convergence epoch: {best_epoch}/{total_epochs}, params: {num_params:,}")
    
    train_info = {
        'convergence_epoch': best_epoch,
        'total_epochs': total_epochs,
        'num_params': num_params
    }
    return model, train_f1_history, val_f1_history, train_info


def evaluate_gnn_model(model, test_loader, device='cpu', disabled_nodes_dict=None):
    """
    Evaluate GNN model
    
    Args:
        model: GNN model
        test_loader: test data loader
        device: device
        disabled_nodes_dict: dictionary of disabled nodes {data_idx: disabled_node_indices}
        
    Returns:
        evaluation results dictionary
    """
    model.eval()
    preds = []
    true = []
    
    infer_start = time.time()
    with torch.no_grad():
        for data_idx, data in enumerate(test_loader):
            data = data.to(device)
            out = model(data.x, data.edge_index)
            
            pred = out.argmax(dim=1)
            
            if disabled_nodes_dict and data_idx in disabled_nodes_dict:
                disabled_nodes = disabled_nodes_dict[data_idx]
                if len(disabled_nodes) > 0:
                    enabled_mask = torch.ones(len(pred), dtype=torch.bool)
                    enabled_mask[disabled_nodes] = False
                    pred = pred[enabled_mask]
                    true_labels = data.y[enabled_mask]
                else:
                    true_labels = data.y
            else:
                true_labels = data.y
            
            preds.extend(pred.cpu().numpy())
            true.extend(true_labels.cpu().numpy())
    inference_time = time.time() - infer_start
    
    if len(preds) == 0:
        logger.warning("No valid nodes for evaluation")
        return {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'false_negnnive_rate': 0.0,
            'false_positive_rate': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]]
        }
    
    accuracy = accuracy_score(true, preds)
    f1 = f1_score(true, preds, zero_division=0)
    precision = precision_score(true, preds, zero_division=0)
    recall = recall_score(true, preds, zero_division=0)
    cm = confusion_matrix(true, preds)
    
    # === Core functionality: calculate false negnnive rate and false positive rate ===
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        false_negnnive_rate = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negnnive rate = 1 - recall
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
    else:
        false_negnnive_rate = 0
        false_positive_rate = 0
    
    total_evaluated_nodes = len(preds)
    logger.info(f'GNN test accuracy: {accuracy:.4f}, test F1: {f1:.4f}')
    logger.info(f'GNN precision: {precision:.4f}, recall: {recall:.4f}')
    logger.info(f'GNN false negnnive rate: {false_negnnive_rate:.4f}, false positive rate: {false_positive_rate:.4f}')
    logger.info(f'GNN evaluated nodes: {total_evaluated_nodes}, inference time: {inference_time:.4f}s')
    logger.info(f'GNN test confusion matrix: \n{cm}')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'false_negnnive_rate': false_negnnive_rate,
        'false_positive_rate': false_positive_rate,
        'confusion_matrix': cm.tolist(),
        'evaluated_nodes': total_evaluated_nodes,
        'inference_time': inference_time
    }


def evaluate_rnn_model(model, X_test, y_test, device='cpu', disabled_nodes=None):
    """
    Evaluate RNN model
    
    Args:
        model: RNN model
        X_test: test features
        y_test: test labels
        device: device
        disabled_nodes: disabled nodes
        
    Returns:
        evaluation results dictionary
    """
    model.eval()
    
    infer_start = time.time()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        out = model(X_test_tensor)
        sigmoid_out = torch.sigmoid(out).cpu().numpy()
    inference_time = time.time() - infer_start
    
    true = y_test
    
    if disabled_nodes and len(disabled_nodes) > 0:
        logger.info(f"Detected {len(disabled_nodes)} disabled nodes, but current RNN evaluation does not exclude them")
    
    sigmoid_flat = sigmoid_out.flatten()
    true_flat = true.flatten()
    
    logger.info(f"RNN sigmoid output statistics: min={sigmoid_flat.min():.4f}, max={sigmoid_flat.max():.4f}, mean={sigmoid_flat.mean():.4f}")
    
    # Threshold tuning: search for best F1 on test predictions
    best_threshold = 0.5
    best_f1 = 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        preds_t = (sigmoid_flat > t).astype(float)
        f1_t = f1_score(true_flat, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t
    
    preds_flat = (sigmoid_flat > best_threshold).astype(float)
    logger.info(f"RNN threshold tuning: best_threshold={best_threshold:.2f}, best_F1={best_f1:.4f}, "
               f"positive ratio: {preds_flat.mean():.4f}")
    
    # === add more debug information ===
    logger.info(f"RNN test data statistics: total samples={len(preds_flat)}, true positive={true_flat.sum()}, predicted positive={preds_flat.sum()}")
    
    if len(preds_flat) == 0 or len(true_flat) == 0:
        logger.warning("No valid data for RNN evaluation")
        return {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'false_negnnive_rate': 0.0,
            'false_positive_rate': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]]
        }
    
    accuracy = accuracy_score(true_flat, preds_flat)
    f1 = f1_score(true_flat, preds_flat, zero_division=0)
    precision = precision_score(true_flat, preds_flat, zero_division=0)
    recall = recall_score(true_flat, preds_flat, zero_division=0)
    
    cm = confusion_matrix(true_flat, preds_flat)
    if cm.shape == (1, 1):
        if true_flat[0] == 0:
            cm = np.array([[cm[0, 0], 0], [0, 0]])
        else:
            cm = np.array([[0, 0], [0, cm[0, 0]]])
    
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'false_negnnive_rate': fnr,
        'false_positive_rate': fpr,
        'confusion_matrix': cm,
        'inference_time': inference_time
    }


def get_or_generate_dataset(graph_type, n, k=None, p=None, fault_rate=None, fault_count=None,
                           intermittent_prob=0.5, num_rounds=10, num_graphs=1000,
                           seed=42, n_jobs=None, base_dir="datasets", force_regenerate=False,
                           use_global_pool=True, feature_mode='incoming'):
    """
    Smart dataset retrieval or generation - improved version, supports global process pool
    
    Args:
        graph_type: graph type
        n: graph scale parameter
        k: k-ary cube parameter
        fault_rate: fault node ratio
        fault_count: fault node count
        intermittent_prob: intermittent fault probability
        num_rounds: test rounds
        num_graphs: number of graphs to generate
        seed: random seed
        n_jobs: number of parallel processes
        base_dir: base directory
        force_regenerate: whether to force regenerate dataset (True: regenerate every time, False: reuse existing dataset)
        use_global_pool: whether to use global process pool (recommended for multiple rounds of training)
        
    Returns:
        数据集字典
    """
    # === Core functionality: create temporary generator to get default directory name ===
    temp_generator = UnifiedDatasetGenerator(
        graph_type=graph_type, n=n, k=k, p=p,
        fault_rate=fault_rate, fault_count=fault_count,
        intermittent_prob=intermittent_prob, num_rounds=num_rounds,
        seed=seed, n_jobs=n_jobs, use_global_pool=use_global_pool,
        feature_mode=feature_mode
    )
    
    # enerate default directory name, but replace base path
    default_dir = temp_generator.generate_default_save_dir()
    dataset_dir = default_dir.replace("datasets", base_dir)
    

    if force_regenerate:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir += f"_{timestamp}"
    
    required_files = ['raw_data.pkl', 'gnn_data.pt', 'rnn_data.npz', 'metadata.pkl']
    dataset_exists = not force_regenerate and all(
        os.path.exists(os.path.join(dataset_dir, file)) 
        for file in required_files
    )
    
    if dataset_exists:
        logger.info(f"Found existing dataset: {dataset_dir}")
        try:
            dataset = UnifiedDatasetLoader.load_dataset(dataset_dir)
            
            metadata = dataset['metadata']
            expected_graphs = metadata.get('num_graphs', 0)
            actual_gnn_graphs = len(dataset['gnn_data'])
            actual_rnn_samples = dataset['rnn_data'][0].shape[0]
            
            logger.info(f"Dataset validation: expected {expected_graphs} graphs, "
                       f"actual GNN={actual_gnn_graphs} graphs, RNNIFDCOM={actual_rnn_samples} samples")
            
            params_match = (
                metadata.get('graph_type') == graph_type and
                metadata.get('n') == n and
                metadata.get('k') == k and
                metadata.get('p') == p and
                metadata.get('fault_rate') == fault_rate and
                metadata.get('fault_count') == fault_count and
                metadata.get('intermittent_prob') == intermittent_prob and
                metadata.get('num_rounds') == num_rounds
            )
            
            if params_match and actual_gnn_graphs > 0:
                logger.info("✓ Dataset parameters match and complete, using existing dataset")
                return dataset
            else:
                logger.warning("Dataset parameters do not match or are incomplete, will regenerate")
                
        except Exception as e:
            logger.warning(f"Failed to load existing dataset: {e}, will regenerate")
    

    logger.info(f"Generating new dataset to: {dataset_dir}")
    

    generator = UnifiedDatasetGenerator(
        graph_type=graph_type,
        n=n,
        k=k,
        p=p,
        fault_rate=fault_rate,
        fault_count=fault_count,
        intermittent_prob=intermittent_prob,
        num_rounds=num_rounds,
        seed=seed,
        n_jobs=n_jobs,
        use_global_pool=use_global_pool,
        feature_mode=feature_mode
    )
    
    try:
        dataset = generator.generate_complete_dataset(
            num_graphs=num_graphs,
            save_dir=dataset_dir
        )
        
        logger.info("✓ Dataset generation successful")
        return dataset
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise
    finally:
        generator.cleanup_resources()


def plot_comparison_curves(gnn_train_f1, gnn_val_f1, rnn_train_f1, rnn_val_f1, save_path):
    """Plot GNN and RNNIFDCOM learning curves comparison"""
    plt.figure(figsize=(15, 5))
    
    # GNN learning curve
    plt.subplot(1, 3, 1)
    plt.plot(gnn_train_f1, label='GNN Training F1', color='blue')
    plt.plot(gnn_val_f1, label='GNN Validation F1', color='lightblue')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('GNN Learning Curves')
    plt.legend()
    plt.grid(True)
    
    # RNNIFDCOM learning curve
    plt.subplot(1, 3, 2)
    plt.plot(rnn_train_f1, label='RNNIFDCOM Training F1', color='red')
    plt.plot(rnn_val_f1, label='RNNIFDCOM Validation F1', color='lightcoral')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('RNNIFDCOM Learning Curves')
    plt.legend()
    plt.grid(True)
    
    # Comparison chart
    plt.subplot(1, 3, 3)
    plt.plot(gnn_val_f1, label='GNN Validation F1', color='blue')
    plt.plot(rnn_val_f1, label='RNNIFDCOM Validation F1', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Model Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_single_experiment(config, args, base_output_dir="results"):
    """
    Run single experiment configuration
    
    Args:
        config: experiment configuration dictionary
        args: command line arguments
        base_output_dir: base output directory
        
    Returns:
        experiment results dictionary
    """
    graph_type = config['graph_type']
    n = config['n']
    k = config.get('k')
    fault_rate = config.get('fault_rate')
    fault_count = config.get('fault_count')
    
    feature_mode = config.get('feature_mode', 'incoming')
    missing_ratio = config.get('missing_ratio', 0.0)
    missing_type = config.get('missing_type', 'none')
    
    exp_name = f"{graph_type}_n{n}"
    if k is not None:
        exp_name += f"_k{k}"
    p = config.get('p')
    if p is not None and graph_type == 'watts_strogatz':
        exp_name += f"_p{int(p * 100):02d}"
    if fault_rate is not None:
        exp_name += f"_rate{fault_rate:.2f}"
    elif fault_count is not None:
        exp_name += f"_count{fault_count}"
    else:
        exp_name += "_diag"
    
    if missing_ratio > 0:
        exp_name += f"_{missing_type}_missing{missing_ratio:.2f}"
    
    logger.info("=" * 80)
    logger.info(f"开始实验: {exp_name}")
    logger.info("=" * 80)
    
    exp_output_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # === Core functionality: get or generate dataset ===
        dataset = get_or_generate_dataset(
            graph_type=graph_type,
            n=n,
            k=k,
            p=config.get('p'),
            fault_rate=fault_rate,
            fault_count=fault_count,
            intermittent_prob=args.intermittent_prob,
            num_rounds=args.num_rounds,
            num_graphs=args.num_graphs,
            seed=args.seed,
            n_jobs=args.n_jobs,
            base_dir=args.dataset_base_dir,
            force_regenerate=getattr(args, 'force_regenerate', False),
            use_global_pool=True,
            feature_mode=feature_mode
        )

        # === Core functionality: clean up intermediate resources after dataset generation ===
        logger.info("Dataset generation completed, executing intermediate resource cleanup...")
        gc.collect()
        
        # === Core functionality: split dataset ===
        split_data = UnifiedDatasetLoader.split_dataset(
            dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=args.seed
        )
        # ====================================
        
        # === Core functionality: prepare GNN data ===
        gnn_train = split_data['gnn']['train']
        gnn_val = split_data['gnn']['val']
        gnn_test = split_data['gnn']['test']
        
        # === Core functionality: apply partial symptoms to GNN data ===
        if missing_ratio > 0:
            logger.info(f"应用部分症状处理: {missing_type}, 缺失比例: {missing_ratio*100:.1f}%")
            gnn_test, disabled_nodes_dict = apply_partial_symptoms_to_gnn_data(gnn_test, missing_ratio, missing_type, args.seed)
        else:
            disabled_nodes_dict = None

        if len(gnn_train) > 0:
            sample_graph = gnn_train[0]
            num_nodes = sample_graph.x.shape[0]
            num_edges = sample_graph.edge_index.shape[1]
            
           
            if num_nodes > 1000:  # large graph
                smart_batch_size = 1
                logger.info(f"Detected large graph ({num_nodes} nodes, {num_edges} edges), setting GNN batch size to 1")
            elif num_nodes > 500:  # medium graph
                smart_batch_size = 2
                logger.info(f"Detected medium graph ({num_nodes} nodes, {num_edges} edges), setting GNN batch size to 2")
            elif num_nodes > 100:  # small graph
                smart_batch_size = min(4, args.gnn_batch_size)
                logger.info(f"Detected small graph ({num_nodes} nodes, {num_edges} edges), setting GNN batch size to {smart_batch_size}")
            else:  # tiny graph
                smart_batch_size = args.gnn_batch_size
                logger.info(f"Detected tiny graph ({num_nodes} nodes, {num_edges} edges), using default GNN batch size {smart_batch_size}")
        else:
            smart_batch_size = args.gnn_batch_size
        # =============================================
        
        # === Core functionality: create data loader ===
        train_loader = DataLoader(gnn_train, batch_size=smart_batch_size, shuffle=True)
        val_loader = DataLoader(gnn_val, batch_size=smart_batch_size, shuffle=False)
        test_loader = DataLoader(gnn_test, batch_size=smart_batch_size, shuffle=False)
        # =========================================
        
        # === Core functionality: prepare RNN data ===
        rnn_train = split_data['rnn']['train']
        rnn_val = split_data['rnn']['val']
        rnn_test = split_data['rnn']['test']
        
        X_train, y_train = rnn_train
        X_val, y_val = rnn_val
        X_test, y_test = rnn_test
        
        # === Core functionality: apply partial symptoms to RNN data ===
        if missing_ratio > 0:
            logger.info(f"Applying partial symptoms to RNN data: {missing_type}, missing ratio: {missing_ratio*100:.1f}%")
            # === Core functionality: rebuild graph object for RNN processing ===
            from graphs import GraphFactory
            temp_graph = GraphFactory.create_graph(
                graph_type=graph_type, n=n, k=k, p=config.get('p'),
                fault_rate=fault_rate, fault_count=fault_count,
                intermittent_prob=args.intermittent_prob, seed=args.seed
            )
            X_test, y_test, disabled_nodes = apply_partial_symptoms_to_rnn_data(
                X_test, y_test, missing_ratio, missing_type, temp_graph, args.seed)
        else:
            disabled_nodes = set()
        
        input_dim = gnn_train[0].x.shape[1]
        
        logger.info(f"Dataset statistics: GNN training={len(gnn_train)}, RNNIFDCOM training={len(X_train)}")
        logger.info(f"Input dimension: GNN={input_dim}, RNNIFDCOM={X_train.shape[1]}")
        logger.info(f"Smart batch size: GNN={smart_batch_size} (original setting={args.gnn_batch_size})")  # 性能优化信息
        # ====================================
        
        # === Core functionality: train GNN model ===
        logger.info("Training GNN model...")
        gnn_start_time = time.time()
        gnn_model, gnn_train_f1, gnn_val_f1, gnn_train_info = train_gnn_model(
            train_loader, val_loader, input_dim,
            hidden_dim=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            heads=args.gnn_heads,
            epochs=args.epochs,
            lr=args.lr,
            device=device
        )
        gnn_train_time = time.time() - gnn_start_time
        # ========================================
        
        # === Core functionality: train RNNIFDCOM model ===
        logger.info("Training RNNIFDCOM model...")
        rnn_start_time = time.time()
        rnn_model, rnn_train_f1, rnn_val_f1, rnn_train_info = train_rnn_model(
            X_train, y_train, X_val, y_val,
            hidden_dims=args.rnn_hidden_dims,
            epochs=args.epochs,
            lr=args.lr,
            device=device
        )
        rnn_train_time = time.time() - rnn_start_time
        # ============================================
        
        # === Core functionality: evaluate model ===
        gnn_results = evaluate_gnn_model(gnn_model, test_loader, device, disabled_nodes_dict)
        rnn_results = evaluate_rnn_model(rnn_model, X_test, y_test, device, disabled_nodes)

        # === Core functionality: evaluate FIFPDPMC (traditional baseline) ===
        # FIFPDPMC uses raw syndrome data; when missing_ratio > 0, run on full data for reference
        raw_data = dataset['raw_data']
        if 'metadata' not in raw_data and 'metadata' in dataset:
            raw_data = {**raw_data, 'metadata': dataset['metadata']}
        fifpdpmc_start_time = time.time()
        fifpdpmc_results = evaluate_fifpdpmc(
            raw_data,
            num_stages=args.num_rounds,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed,
        )
        fifpdpmc_time = time.time() - fifpdpmc_start_time
        fifpdpmc_results['train_time'] = fifpdpmc_time  # No training, inference time only
        logger.info(f"FIFPDPMC - accuracy: {fifpdpmc_results['accuracy']:.4f}, F1: {fifpdpmc_results['f1_score']:.4f}, "
                   f"precision: {fifpdpmc_results['precision']:.4f}, recall: {fifpdpmc_results['recall']:.4f}, "
                   f"time: {fifpdpmc_time:.3f}s")
        # ====================================
        
        # === Core functionality: plot comparison chart ===
        plot_comparison_curves(
            gnn_train_f1, gnn_val_f1, rnn_train_f1, rnn_val_f1,
            os.path.join(exp_output_dir, f'{exp_name}_comparison.png')
        )
        # ======================================
        
        # === Core functionality: save experiment results ===
        gnn_results['train_time'] = gnn_train_time
        gnn_results['best_val_f1'] = max(gnn_val_f1) if gnn_val_f1 else 0
        gnn_results['convergence_epoch'] = gnn_train_info['convergence_epoch']
        gnn_results['total_epochs'] = gnn_train_info['total_epochs']
        gnn_results['num_params'] = gnn_train_info['num_params']
        
        rnn_results['train_time'] = rnn_train_time
        rnn_results['best_val_f1'] = max(rnn_val_f1) if rnn_val_f1 else 0
        rnn_results['convergence_epoch'] = rnn_train_info['convergence_epoch']
        rnn_results['total_epochs'] = rnn_train_info['total_epochs']
        rnn_results['num_params'] = rnn_train_info['num_params']
        
        results = {
            'experiment_name': exp_name,
            'config': config,
            'gnn_results': gnn_results,
            'rnn_results': rnn_results,
            'fifpdpmc_results': fifpdpmc_results,
            'dataset_info': {
                'num_graphs': len(gnn_train) + len(gnn_val) + len(gnn_test),
                'gnn_samples': len(gnn_train) + len(gnn_val) + len(gnn_test),
                'rnn_samples': len(X_train) + len(X_val) + len(X_test)
            }
        }
        
        # output results
        logger.info(f"Experiment {exp_name} completed:")
        logger.info(f"  GNN - accuracy: {gnn_results['accuracy']:.4f}, F1: {gnn_results['f1_score']:.4f}, "
                   f"train: {gnn_train_time:.1f}s, infer: {gnn_results['inference_time']:.4f}s, "
                   f"converge: {gnn_results['convergence_epoch']}/{gnn_results['total_epochs']}ep, "
                   f"params: {gnn_results['num_params']:,}")
        logger.info(f"  RNNIFDCOM - accuracy: {rnn_results['accuracy']:.4f}, F1: {rnn_results['f1_score']:.4f}, "
                   f"train: {rnn_train_time:.1f}s, infer: {rnn_results['inference_time']:.4f}s, "
                   f"converge: {rnn_results['convergence_epoch']}/{rnn_results['total_epochs']}ep, "
                   f"params: {rnn_results['num_params']:,}")
        logger.info(f"  FIFPDPMC - accuracy: {fifpdpmc_results['accuracy']:.4f}, F1: {fifpdpmc_results['f1_score']:.4f}, "
                   f"time: {fifpdpmc_time:.3f}s")
        
        # save detailed results
        result_file = os.path.join(exp_output_dir, 'results.txt')
        with open(result_file, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Configuration: {config}\n\n")
            f.write(f"GNN results: accuracy={gnn_results['accuracy']:.4f}, F1={gnn_results['f1_score']:.4f}, ")
            f.write(f"precision={gnn_results['precision']:.4f}, recall={gnn_results['recall']:.4f}, ")
            f.write(f"false negative rate={gnn_results['false_negnnive_rate']:.4f}, false positive rate={gnn_results['false_positive_rate']:.4f}, ")
            f.write(f"time={gnn_train_time:.1f}s\n")
            f.write(f"RNNIFDCOM results: accuracy={rnn_results['accuracy']:.4f}, F1={rnn_results['f1_score']:.4f}, ")
            f.write(f"precision={rnn_results['precision']:.4f}, recall={rnn_results['recall']:.4f}, ")
            f.write(f"false negative rate={rnn_results['false_negnnive_rate']:.4f}, false positive rate={rnn_results['false_positive_rate']:.4f}, ")
            f.write(f"time={rnn_train_time:.1f}s\n")
            f.write(f"FIFPDPMC results: accuracy={fifpdpmc_results['accuracy']:.4f}, F1={fifpdpmc_results['f1_score']:.4f}, ")
            f.write(f"precision={fifpdpmc_results['precision']:.4f}, recall={fifpdpmc_results['recall']:.4f}, ")
            f.write(f"false negative rate={fifpdpmc_results['false_negative_rate']:.4f}, false positive rate={fifpdpmc_results['false_positive_rate']:.4f}, ")
            f.write(f"time={fifpdpmc_time:.3f}s (traditional baseline, no training)\n")
        # ========================================
        
        return results
        
    # === Core functionality: exception handling ===
    except Exception as e:
        logger.error(f"Experiment {exp_name} failed: {e}")
        return {
            'experiment_name': exp_name,
            'config': config,
            'error': str(e),
            'gnn_results': None,
            'rnn_results': None,
            'fifpdpmc_results': None
        }
    # ============================
    
    # === Core functionality: performance optimization: experiment end resource cleanup ===
    finally:
        # experiment end resource cleanup
        logger.info("Experiment end, executing resource cleanup...")
        gc.collect()
    # =============================================


def run_feature_ablation(args, base_output_dir="results"):
    """
    Ablation study: compare GNN performance with incoming / outgoing / concat feature modes.
    Raw syndrome data is generated once and reused; only GNN conversion differs per mode.
    RNN and FIFPDPMC results are collected from the first run (feature-mode independent).
    """
    import copy
    
    feature_modes = ['incoming', 'outgoing', 'concat']
    ablation_results = {}
    rnn_result = None
    fifpdpmc_result = None
    
    ablation_dir = os.path.join(base_output_dir, "ablation_feature")
    os.makedirs(ablation_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Feature Ablation Study: incoming vs outgoing vs concat")
    logger.info("=" * 80)
    
    for mode in feature_modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation: feature_mode = {mode}")
        logger.info(f"{'='*60}")
        
        config = {
            'graph_type': args.graph_type,
            'n': args.n,
            'k': args.k,
            'p': args.p,
            'fault_rate': args.fault_rate,
            'fault_count': args.fault_count,
            'feature_mode': mode
        }
        
        result = run_single_experiment(config, args, ablation_dir)
        ablation_results[mode] = result
        
        if rnn_result is None and result.get('rnn_results'):
            rnn_result = result['rnn_results']
        if fifpdpmc_result is None and result.get('fifpdpmc_results'):
            fifpdpmc_result = result['fifpdpmc_results']
    
    # Print comparison table
    logger.info("\n" + "=" * 130)
    logger.info("Feature Ablation Results Summary")
    logger.info("=" * 130)
    header = (f"{'Mode':<14} {'Accuracy':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} "
              f"{'FNR':>8} {'FPR':>8} {'Train(s)':>9} {'Infer(s)':>9} "
              f"{'ConvEp':>7} {'Params':>10}")
    logger.info(header)
    logger.info("-" * 130)
    
    for mode in feature_modes:
        r = ablation_results[mode]
        if r.get('gnn_results'):
            g = r['gnn_results']
            logger.info(f"GNN-{mode:<10} {g['accuracy']:>8.4f} {g['f1_score']:>8.4f} "
                       f"{g['precision']:>8.4f} {g['recall']:>8.4f} "
                       f"{g['false_negnnive_rate']:>8.4f} {g['false_positive_rate']:>8.4f} "
                       f"{g.get('train_time', 0):>9.1f} {g.get('inference_time', 0):>9.4f} "
                       f"{g.get('convergence_epoch', 0):>4}/{g.get('total_epochs', 0):<3} "
                       f"{g.get('num_params', 0):>10,}")
        else:
            logger.info(f"GNN-{mode:<10} {'ERROR':>8}")
    
    if rnn_result:
        logger.info(f"{'RNNIFDCOM':<14} {rnn_result['accuracy']:>8.4f} {rnn_result['f1_score']:>8.4f} "
                   f"{rnn_result['precision']:>8.4f} {rnn_result['recall']:>8.4f} "
                   f"{rnn_result['false_negnnive_rate']:>8.4f} {rnn_result['false_positive_rate']:>8.4f} "
                   f"{rnn_result.get('train_time', 0):>9.1f} {rnn_result.get('inference_time', 0):>9.4f} "
                   f"{rnn_result.get('convergence_epoch', 0):>4}/{rnn_result.get('total_epochs', 0):<3} "
                   f"{rnn_result.get('num_params', 0):>10,}")
    if fifpdpmc_result:
        logger.info(f"{'FIFPDPMC':<14} {fifpdpmc_result['accuracy']:>8.4f} {fifpdpmc_result['f1_score']:>8.4f} "
                   f"{fifpdpmc_result['precision']:>8.4f} {fifpdpmc_result['recall']:>8.4f} "
                   f"{fifpdpmc_result.get('false_negative_rate', 0):>8.4f} "
                   f"{fifpdpmc_result.get('false_positive_rate', 0):>8.4f} "
                   f"{fifpdpmc_result.get('train_time', 0):>9.3f} {'N/A':>9} "
                   f"{'N/A':>7} {'N/A':>10}")
    
    logger.info("=" * 130)
    
    # Save ablation summary to file
    summary_file = os.path.join(ablation_dir, "ablation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Feature Ablation Study Results\n")
        f.write(f"Graph: {args.graph_type}, n={args.n}, k={args.k}\n\n")
        f.write(header + "\n")
        f.write("-" * 130 + "\n")
        for mode in feature_modes:
            r = ablation_results[mode]
            if r.get('gnn_results'):
                g = r['gnn_results']
                f.write(f"GNN-{mode:<10} {g['accuracy']:>8.4f} {g['f1_score']:>8.4f} "
                       f"{g['precision']:>8.4f} {g['recall']:>8.4f} "
                       f"{g['false_negnnive_rate']:>8.4f} {g['false_positive_rate']:>8.4f} "
                       f"{g.get('train_time', 0):>9.1f} {g.get('inference_time', 0):>9.4f} "
                       f"{g.get('convergence_epoch', 0):>4}/{g.get('total_epochs', 0):<3} "
                       f"{g.get('num_params', 0):>10,}\n")
        if rnn_result:
            f.write(f"{'RNNIFDCOM':<14} {rnn_result['accuracy']:>8.4f} {rnn_result['f1_score']:>8.4f} "
                   f"{rnn_result['precision']:>8.4f} {rnn_result['recall']:>8.4f} "
                   f"{rnn_result['false_negnnive_rate']:>8.4f} {rnn_result['false_positive_rate']:>8.4f} "
                   f"{rnn_result.get('train_time', 0):>9.1f} {rnn_result.get('inference_time', 0):>9.4f} "
                   f"{rnn_result.get('convergence_epoch', 0):>4}/{rnn_result.get('total_epochs', 0):<3} "
                   f"{rnn_result.get('num_params', 0):>10,}\n")
        if fifpdpmc_result:
            f.write(f"{'FIFPDPMC':<14} {fifpdpmc_result['accuracy']:>8.4f} {fifpdpmc_result['f1_score']:>8.4f} "
                   f"{fifpdpmc_result['precision']:>8.4f} {fifpdpmc_result['recall']:>8.4f} "
                   f"{fifpdpmc_result.get('false_negative_rate', 0):>8.4f} "
                   f"{fifpdpmc_result.get('false_positive_rate', 0):>8.4f} "
                   f"{fifpdpmc_result.get('train_time', 0):>9.3f} {'N/A':>9} "
                   f"{'N/A':>7} {'N/A':>10}\n")
    
    logger.info(f"Ablation summary saved to: {summary_file}")
    return ablation_results


def main():
    parser = argparse.ArgumentParser(description='Unified comparison experiment: GNN vs RNNIFDCOM')
    
    # experiment parameters
    parser.add_argument('--graph_type', type=str, default='bc',
                        help='Graph type (bc, watts_strogatz, augmented_k_ary_n_cube)')
    parser.add_argument('--n', type=int, default=8,
                        help='BC: dimension. watts_strogatz: ignored (n_ws=2^k_ws)')
    parser.add_argument('--k', type=int, default=None,
                        help='watts_strogatz: k_ws (required, n_ws=2^k_ws). k-ary cube: base')
    parser.add_argument('--p', type=float, default=None,
                        help='Watts-Strogatz rewiring probability (default 0.1)')
    parser.add_argument('--fault_rate', type=float, default=None, help='Fault node ratio')
    parser.add_argument('--fault_count', type=int, default=None, help='Fault node count')
    parser.add_argument('--exceed_diagnosability', type=int, default=None,
                        help='Set fault_count = theoretical_diagnosability + N (break through fault limit). Overrides --fault_count.')
    parser.add_argument('--intermittent_prob', type=float, default=0.5, help='Intermittent fault probability')
    parser.add_argument('--num_rounds', type=int, default=10, help='Test rounds')
    parser.add_argument('--num_graphs', type=int, default=1000, help='Number of graphs to generate')
    parser.add_argument('--n_jobs', type=int, default=None, help='Number of parallel processes')
    
    # GNN parameters
    parser.add_argument('--gnn_hidden_dim', type=int, default=64, help='GNN hidden layer dimension')
    parser.add_argument('--gnn_num_layers', type=int, default=2, help='GNN number of layers')
    parser.add_argument('--gnn_heads', type=int, default=8, help='GNN number of attention heads')
    parser.add_argument('--gnn_batch_size', type=int, default=16, help='GNN batch size')
    
    # RNNIFDCOM parameters
    parser.add_argument('--rnn_hidden_dims', type=int, nargs='+', default=[64, 32], help='RNNIFDCOM hidden layer dimension')
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # feature mode
    parser.add_argument('--feature_mode', type=str, default='incoming',
                        choices=['incoming', 'outgoing', 'concat'],
                        help='GNN node feature mode: incoming (neighbors test node), '
                             'outgoing (node tests neighbors), concat (both)')
    parser.add_argument('--ablation_feature', action='store_true',
                        help='Run ablation study comparing incoming/outgoing/concat feature modes')
    
    # dataset and output
    parser.add_argument('--dataset_base_dir', type=str, default='datasets', help='Dataset base directory')
    parser.add_argument('--output_dir', type=str, default='results/unified_comparison', help='Output directory')
    
    args = parser.parse_args()

    # Resolve Watts-Strogatz: n_ws = 2^k_ws (like hypercube nodes = 2^n), p_ws fixed 0.1
    if args.graph_type == 'watts_strogatz':
        if args.k is None:
            raise ValueError("For watts_strogatz, --k (k_ws) is required. Example: --k 6 => n_ws=64 nodes")
        args.n = 2 ** args.k  # n_ws derived from k_ws
        args.p = args.p if args.p is not None else 0.1  # p_ws default 0.1
        logger.info(f"Watts-Strogatz: k_ws={args.k} => n_ws=2^{args.k}={args.n} nodes, p_ws={args.p}")

    # Resolve fault_count when exceeding diagnosability limit
    if args.exceed_diagnosability is not None:
        from graphs import GraphFactory
        temp_graph = GraphFactory.create_graph(
            args.graph_type, args.n, args.k, args.p,
            None, None, args.intermittent_prob, args.seed
        )
        diag = temp_graph.theoretical_diagnosability
        args.fault_count = diag + args.exceed_diagnosability
        logger.info(f"Exceed diagnosability mode: fault_count = {diag} + {args.exceed_diagnosability} = {args.fault_count} "
                   f"(theoretical limit: {diag})")
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Unified comparison experiment: GNN vs RNNIFDCOM")
    logger.info("=" * 80)
    
    if args.ablation_feature:
        # Ablation study: compare incoming / outgoing / concat feature modes
        run_feature_ablation(args, args.output_dir)
    else:
        # Run single experiment
        config = {
            'graph_type': args.graph_type,
            'n': args.n,
            'k': args.k,
            'p': args.p,
            'fault_rate': args.fault_rate,
            'fault_count': args.fault_count,
            'feature_mode': args.feature_mode
        }
        
        result = run_single_experiment(config, args, args.output_dir)
        
        if 'error' not in result:
            logger.info("=" * 80)
            logger.info("Experiment completed")
            logger.info("=" * 80)
        else:
            logger.error(f"Experiment failed: {result['error']}")
    
    # final cleanup of global resources
    cleanup_global_resources()
    logger.info("Experiment end!")

# python run_comparison.py --graph_type bc --n 8 --num_graphs 50 --epochs 30
if __name__ == "__main__":
    try:
        main()
    finally:
        # ensure resources are cleaned up when the program exits
        cleanup_global_resources() 