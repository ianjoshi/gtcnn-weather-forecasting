from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GraphNorm


class GTCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 3,
        num_layers: int = 3,
        dropout: float = 0.1,
        residual: bool = True,
        use_bn: bool = True,
        incremental_k: bool = False,  
    ) -> None:
        """
        Args:
            in_channels: node feature dimension (C_in)
            hidden_channels: width of hidden layers
            out_channels: target dimension (C_out)
            K: base Chebyshev order (used if incremental_k=False)
            num_layers: number of ChebConv blocks
            dropout: dropout probability after each block
            residual: add residual skip where shapes match
            use_bn: use BatchNorm1d on node features
            incremental_k: if True, layer i uses K=i (starting from 1 hop)
        """
        super().__init__()
        assert num_layers >= 2, "Use at least 2 layers for non-trivial receptive field."

        self.dropout = dropout
        self.residual = residual
        self.use_bn = use_bn
        self.incremental_k = incremental_k

        layers = []
        bns = []

        # Determine K for each layer
        # If incremental_k=True : k_hops = [1, 2, 3, ...], else : k_hops = [K, K, K, ...]
        k_hops = [i + 1 for i in range(num_layers)] if incremental_k else [K] * num_layers

        # First layer
        layers.append(ChebConv(in_channels, hidden_channels, K=k_hops[0], normalization="sym"))
        if use_bn:
            bns.append(GraphNorm(hidden_channels))

        # Hidden layers
        for layer_idx in range(1, num_layers - 1):
            layers.append(ChebConv(hidden_channels, hidden_channels, K=k_hops[layer_idx], normalization="sym"))
            if use_bn:
                bns.append(GraphNorm(hidden_channels))

        # Final layer
        layers.append(ChebConv(hidden_channels, out_channels, K=k_hops[-1], normalization="sym"))

        self.layers = nn.ModuleList(layers)
        self.bns = nn.ModuleList(bns)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, N: int, T: int) -> torch.Tensor:
        """
        x: [N*T, C_in] flattened over time then space 
        edge_index: [2, E]
        N: nodes per time slice (H*W)
        T: number of time steps in the input sequence
        returns: [N, C_out] predictions for the last time slice only
        """
        X = x
        bn_idx = 0

        for layer_idx, conv_layer in enumerate(self.layers):
            X_in = X  # for residual connection
            X = conv_layer(X, edge_index)

            if layer_idx < len(self.layers) - 1:  # hidden blocks
                if self.use_bn:
                    X = self.bns[bn_idx](X)
                    bn_idx += 1

                X = F.relu(X)
                X = F.dropout(X, p=self.dropout, training=self.training)

                if self.residual and X.shape == X_in.shape:
                    X = X + X_in

        # Output only last time slice
        start = (T - 1) * N
        end = T * N
        y_hat = X[start:end]  # [N, out_channels]

        return y_hat
