"""
GTCNN for your PyG spatio-temporal graph
Note: Uses Chebyshev graph convolution (ChebConv) over the spatio-temporal graph (we can make this more flexible later).
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


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
    ) -> None:
        """
        Args:
            in_channels: node feature dim (C_in)
            hidden_channels: width of hidden layers
            out_channels: target dim (C_out)
            K: Chebyshev order (K=2..3 is usually enough)
            num_layers: number of ChebConv blocks (ideally >=2)
            dropout: dropout probability after each block
            residual: add residual skip where shapes match
            use_bn: use BatchNorm1d on node features
        """
        super().__init__()
        assert num_layers >= 2, "Use at least 2 layers for non-trivial receptive field."

        self.dropout = dropout
        self.residual = residual
        self.use_bn = use_bn

        layers = []
        bns = []

        # first layer
        layers.append(ChebConv(in_channels, hidden_channels, K=K, normalization=None))
        if use_bn:
            bns.append(nn.BatchNorm1d(hidden_channels))

        # hidden layers
        for _ in range(num_layers - 2):
            layers.append(ChebConv(hidden_channels, hidden_channels, K=K, normalization=None))
            if use_bn:
                bns.append(nn.BatchNorm1d(hidden_channels))

        # final layer to out_channels (no BN)
        layers.append(ChebConv(hidden_channels, out_channels, K=K, normalization=None))

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
        h = x
        bn_idx = 0
        for li, conv in enumerate(self.layers):
            h_in = h
            h = conv(h, edge_index)  
            if li < len(self.layers) - 1:  # hidden blocks
                if self.use_bn:
                    h = self.bns[bn_idx](h)
                    bn_idx += 1
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.residual and h.shape == h_in.shape:
                    h = h + h_in

        # Take last time slice only
        start = (T - 1) * N
        end = T * N
        y_hat = h[start:end]  # [N, out_channels]
        
        return y_hat



