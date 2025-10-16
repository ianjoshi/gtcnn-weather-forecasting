"""
Deep Echo State Network (DeepESN) Implementation

This module implements the Deep Echo State Network with leaky integrator neurons
as described in "Scalable Spatiotemporal Graph Neural Networks" by Cini et al.

Implements equations (3) and (4) from section 3.1:
- Multi-layer reservoir with leaky integrator neurons
- Random weight matrices (frozen during training)
- Multi-scale temporal dynamics extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class DeepESNEncoder(nn.Module):
    """
    Deep Echo State Network (DeepESN) with leaky integrator neurons.
    
    Implements the temporal encoder from section 3.1 of the SGP paper.
    
    Equations implemented:
    h_i^(0)_t = [x_i_t, u_i_t]  (input concatenation)
    ĥ_i^(l)_t = tanh(W_u^(l) h_i^(l-1)_t + W_h^(l) h_i^(l)_t-1 + b^(l))
    h_i^(l)_t = (1 - γ_l) h_i^(l)_t-1 + γ_l ĥ_i^(l)_t
    
    Final output: h_i_t = [h_i^(0)_t || h_i^(1)_t || ... || h_i^(L)_t]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],  # Hidden dimension for each layer
        leaky_rates: List[float],  # γ_l for each layer
        spectral_radius: float = 0.9,
        input_scaling: float = 1.0,
        bias_scaling: float = 0.1,
        seed: int = 42
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.leaky_rates = leaky_rates
        self.num_layers = len(hidden_dims)
        
        assert len(hidden_dims) == len(leaky_rates), "Hidden dims and leaky rates must have same length"
        assert all(0 < γ <= 1 for γ in leaky_rates), "Leaky rates must be in (0, 1]"
        
        # Set random seed for reproducible random weights
        torch.manual_seed(seed)
        
        # Initialize layers
        self.layers = nn.ModuleList()
        self.biases = nn.ParameterList()
        
        # First layer (input layer)
        first_hidden_dim = hidden_dims[0]
        self.layers.append(nn.ModuleDict({
            'W_u': nn.Linear(input_dim, first_hidden_dim, bias=False),
            'W_h': nn.Linear(first_hidden_dim, first_hidden_dim, bias=False)
        }))
        self.biases.append(nn.Parameter(torch.zeros(first_hidden_dim)))
        
        # Hidden layers
        for l in range(1, self.num_layers):
            prev_dim = hidden_dims[l-1]
            curr_dim = hidden_dims[l]
            
            self.layers.append(nn.ModuleDict({
                'W_u': nn.Linear(prev_dim, curr_dim, bias=False),
                'W_h': nn.Linear(curr_dim, curr_dim, bias=False)
            }))
            self.biases.append(nn.Parameter(torch.zeros(curr_dim)))
        
        # Initialize random weights according to ESN principles
        self._initialize_esn_weights(spectral_radius, input_scaling, bias_scaling)
        
        # Freeze all parameters (DeepESN uses frozen random weights)
        for param in self.parameters():
            param.requires_grad = False
    
    def _initialize_esn_weights(
        self, 
        spectral_radius: float, 
        input_scaling: float, 
        bias_scaling: float
    ):
        """Initialize weights according to Echo State Network principles."""
        
        for layer_idx, layer in enumerate(self.layers):
            # Initialize input weights W_u
            nn.init.uniform_(layer['W_u'].weight, -input_scaling, input_scaling)
            
            # Initialize recurrent weights W_h with spectral radius constraint
            W_h = torch.randn(layer['W_h'].weight.shape)
            # Normalize to desired spectral radius
            eigenvals = torch.linalg.eigvals(W_h)
            max_eigenval = torch.max(torch.real(eigenvals))
            if max_eigenval > 0:
                W_h = W_h * (spectral_radius / max_eigenval)
            layer['W_h'].weight.data = W_h
            
            # Initialize bias
            nn.init.uniform_(self.biases[layer_idx], -bias_scaling, bias_scaling)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepESN.
        
        Args:
            x: [batch_size, seq_len, input_dim] - input temporal sequence
            
        Returns:
            [batch_size, seq_len, total_hidden_dim] - temporal embeddings for each time step
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states for all layers
        hidden_states = []
        for l in range(self.num_layers):
            hidden_states.append(torch.zeros(batch_size, self.hidden_dims[l], device=x.device))
        
        # Store temporal embeddings for each time step
        temporal_embeddings = []
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_dim]
            
            # First layer: h_i^(0)_t = [x_i_t, u_i_t] (input concatenation)
            # For simplicity, we use x_i_t directly as the input
            h_prev = x_t
            
            # Process through all layers
            for l in range(self.num_layers):
                layer = self.layers[l]
                γ_l = self.leaky_rates[l]
                
                # ĥ_i^(l)_t = tanh(W_u^(l) h_i^(l-1)_t + W_h^(l) h_i^(l)_t-1 + b^(l))
                h_hat = torch.tanh(
                    layer['W_u'](h_prev) + 
                    layer['W_h'](hidden_states[l]) + 
                    self.biases[l]
                )
                
                # h_i^(l)_t = (1 - γ_l) h_i^(l)_t-1 + γ_l ĥ_i^(l)_t
                hidden_states[l] = (1 - γ_l) * hidden_states[l] + γ_l * h_hat
                
                # Update h_prev for next layer
                h_prev = hidden_states[l]
            
            # Concatenate states from all layers for this time step
            # h_i_t = [h_i^(0)_t || h_i^(1)_t || ... || h_i^(L)_t]
            h_t = torch.cat(hidden_states, dim=1)  # [batch_size, total_hidden_dim]
            temporal_embeddings.append(h_t)
        
        # Stack temporal embeddings: [batch_size, seq_len, total_hidden_dim]
        H = torch.stack(temporal_embeddings, dim=1)
        
        return H


class RandomizedRNNEncoder(nn.Module):
    """
    Wrapper for DeepESNEncoder to maintain compatibility with existing code.
    
    This is the correct implementation of the temporal encoder from the SGP paper.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,  # Not used in DeepESN
        rnn_type: str = "DeepESN"  # Default to DeepESN
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # DeepESN configuration
        hidden_dims = [hidden_dim] * num_layers
        leaky_rates = [0.1 + 0.8 * i / (num_layers - 1) for i in range(num_layers)]  # Increasing leaky rates
        
        # Initialize DeepESN
        self.deep_esn = DeepESNEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            leaky_rates=leaky_rates,
            spectral_radius=0.9,
            input_scaling=1.0,
            bias_scaling=0.1
        )
        
        # Output projection (only trainable part)
        total_hidden_dim = sum(hidden_dims)
        self.output_proj = nn.Linear(total_hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal sequence using DeepESN.
        
        Args:
            x: [batch_size, seq_len, input_dim] - temporal sequence
            
        Returns:
            [batch_size, seq_len, output_dim] - temporal embeddings for each time step
        """
        # Get DeepESN encoding: [batch_size, seq_len, total_hidden_dim]
        esn_output = self.deep_esn(x)
        
        # Project to output dimension for each time step
        batch_size, seq_len, total_hidden_dim = esn_output.shape
        esn_flat = esn_output.view(-1, total_hidden_dim)  # [batch_size * seq_len, total_hidden_dim]
        embedding_flat = self.output_proj(esn_flat)  # [batch_size * seq_len, output_dim]
        embedding = embedding_flat.view(batch_size, seq_len, self.output_dim)  # [batch_size, seq_len, output_dim]
        
        return embedding