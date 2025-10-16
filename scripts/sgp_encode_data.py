"""
SGP Data Encoding Module

This module implements the first phase of the Scalable Graph Predictor (SGP) approach:
pre-computing temporal embeddings using a randomized RNN encoder.

Based on: "Scalable Spatiotemporal Graph Neural Networks" by Cini et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import yaml
from typing import Dict, List, Tuple, Optional

from data.dataloader import get_dataloaders
from models.randomized_rnn_encoder import RandomizedRNNEncoder



class SGPDataEncoder:
    """
    SGP Data Encoder for pre-computing temporal embeddings.
    
    This class handles the first phase of SGP training: encoding all temporal
    sequences in the dataset using the randomized RNN encoder.
    """
    
    def __init__(
        self,
        config: Dict,
        model_config: Dict,
        device: torch.device
    ):
        self.config = config
        self.model_config = model_config
        self.device = device
        
        # Extract SGP-specific config
        sgp_config = model_config["graph_based"]["sgp_gtcnn"]
        
        # Initialize DeepESN encoder
        self.encoder = RandomizedRNNEncoder(
            input_dim=config["data"]["num_features"],  # Will be set from dataset
            hidden_dim=sgp_config["rnn_hidden_size"],
            output_dim=sgp_config["temporal_embedding_dim"],
            num_layers=sgp_config["rnn_num_layers"],
            dropout=sgp_config.get("dropout", 0.1),
            rnn_type=sgp_config.get("rnn_type", "DeepESN")
        ).to(device)
        
        # Setup embedding cache
        self.cache_dir = Path(sgp_config.get("embedding_cache_path", "./sgp_embeddings/"))
        self.cache_dir.mkdir(exist_ok=True)
        
    def encode_dataset(
        self,
        dataloader: DataLoader,
        split_name: str,
        save_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode entire dataset and return embeddings.
        
        Args:
            dataloader: PyTorch Geometric DataLoader
            split_name: Name of the split (train/val/test)
            save_cache: Whether to save embeddings to disk
            
        Returns:
            Dictionary with node embeddings for each sample
        """
        print(f"Encoding {split_name} dataset...")
        
        self.encoder.eval()
        embeddings = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Encoding {split_name}")):
                batch = batch.to(self.device)
                
                # Extract temporal sequences for each node
                N = batch.y.size(0)  # Number of spatial nodes
                T = self.config["graph"]["input_length"]
                C_in = batch.x.size(1)  # Input feature dimension
                
                # Reshape to [N, T, C_in] - temporal sequences for each node
                x_temporal = batch.x.view(N, T, C_in)
                
                # Encode temporal sequences: [N, T, temporal_embedding_dim]
                temporal_embeddings = self.encoder(x_temporal)
                
                # For now, use the final time step embedding (can be changed to use all time steps)
                node_embeddings = temporal_embeddings[:, -1, :]  # [N, temporal_embedding_dim]
                
                # Store embeddings (use batch index as key)
                embeddings[f"{split_name}_{batch_idx}"] = {
                    "embeddings": node_embeddings.cpu(),
                    "targets": batch.y.cpu(),
                    "edge_index": batch.edge_index.cpu(),
                    "num_nodes": N
                }
        
        # Save to cache if requested
        if save_cache:
            cache_path = self.cache_dir / f"{split_name}_embeddings.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Saved {split_name} embeddings to {cache_path}")
        
        return embeddings
    
    def load_cached_embeddings(self, split_name: str) -> Optional[Dict]:
        """Load pre-computed embeddings from cache."""
        cache_path = self.cache_dir / f"{split_name}_embeddings.pkl"
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"Loaded cached {split_name} embeddings from {cache_path}")
            return embeddings
        else:
            print(f"No cached embeddings found for {split_name}")
            return None
    
    def encode_all_splits(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        force_recompute: bool = False
    ) -> Tuple[Dict, Dict, Optional[Dict]]:
        """
        Encode all dataset splits.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loader: Test dataloader (optional)
            force_recompute: Force recomputation even if cache exists
            
        Returns:
            Tuple of (train_embeddings, val_embeddings, test_embeddings)
        """
        # Get dataset info for encoder initialization
        sample_batch = next(iter(train_loader))
        C_in = sample_batch.x.size(1)
        
        # Update encoder input dimension if needed
        if self.encoder.input_dim != C_in:
            print(f"Updating encoder input dimension from {self.encoder.input_dim} to {C_in}")
            # Reinitialize encoder with correct input dimension
            sgp_config = self.model_config["graph_based"]["sgp_gtcnn"]
            self.encoder = RandomizedRNNEncoder(
                input_dim=C_in,
                hidden_dim=sgp_config["rnn_hidden_size"],
                output_dim=sgp_config["temporal_embedding_dim"],
                num_layers=sgp_config["rnn_num_layers"],
                dropout=sgp_config.get("dropout", 0.1),
                rnn_type=sgp_config.get("rnn_type", "DeepESN")
            ).to(self.device)
        
        # Encode training data
        if not force_recompute:
            train_embeddings = self.load_cached_embeddings("train")
        else:
            train_embeddings = None
            
        if train_embeddings is None:
            train_embeddings = self.encode_dataset(train_loader, "train")
        
        # Encode validation data
        if not force_recompute:
            val_embeddings = self.load_cached_embeddings("val")
        else:
            val_embeddings = None
            
        if val_embeddings is None:
            val_embeddings = self.encode_dataset(val_loader, "val")
        
        # Encode test data (if provided)
        test_embeddings = None
        if test_loader is not None:
            if not force_recompute:
                test_embeddings = self.load_cached_embeddings("test")
            else:
                test_embeddings = None
                
            if test_embeddings is None:
                test_embeddings = self.encode_dataset(test_loader, "test")
        
        return train_embeddings, val_embeddings, test_embeddings


def main():
    """Main function to encode dataset for SGP training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Encode dataset for SGP training")
    parser.add_argument("--force_recompute", action="store_true", 
                       help="Force recomputation of embeddings")
    parser.add_argument("--config_path", type=str, default="utils/base_config.yaml",
                       help="Path to base config file")
    parser.add_argument("--model_config_path", type=str, default="utils/model_config.yaml",
                       help="Path to model config file")
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        config=config, 
        model_category="graph_based", 
        eval_mode=False
    )
    
    # Initialize encoder
    encoder = SGPDataEncoder(config, model_config, device)
    
    # Encode all splits
    train_embeddings, val_embeddings, _ = encoder.encode_all_splits(
        train_loader=train_loader,
        val_loader=val_loader,
        force_recompute=args.force_recompute
    )
    
    print("Data encoding completed!")
    print(f"Train embeddings: {len(train_embeddings)} samples")
    print(f"Val embeddings: {len(val_embeddings)} samples")


if __name__ == "__main__":
    main()
