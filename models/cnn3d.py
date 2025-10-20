import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_bn: bool = True,
    ) -> None:
        """
        3D CNN for spatiotemporal weather forecasting with ERA5 data
        
        Args:
            in_channels: number of input weather variables
            hidden_channels: base number of hidden channels (will be scaled in different layers)
            out_channels: number of output weather variables to predict
            num_layers: number of convolutional layers
            dropout: dropout probability
            use_bn: whether to use batch normalization
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        
        # Encoder layers - progressively reduce spatial dimensions while increasing channels
        self.encoder_layers = nn.ModuleList()
        self.encoder_bns = nn.ModuleList() if use_bn else None
        
        # First layer
        self.encoder_layers.append(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        if use_bn:
            self.encoder_bns.append(nn.BatchNorm3d(hidden_channels))
        
        # Middle encoder layers - double channels, halve spatial dimensions
        current_channels = hidden_channels
        for i in range(1, num_layers - 1):
            next_channels = current_channels * 2
            self.encoder_layers.append(
                nn.Conv3d(current_channels, next_channels, kernel_size=(3, 3, 3), 
                         stride=(1, 2, 2), padding=(1, 1, 1))  # Reduce spatial dims
            )
            if use_bn:
                self.encoder_bns.append(nn.BatchNorm3d(next_channels))
            current_channels = next_channels
        
        # Bottleneck layer
        self.bottleneck = nn.Conv3d(current_channels, current_channels, 
                                   kernel_size=(3, 3, 3), padding=(1, 1, 1))
        if use_bn:
            self.bottleneck_bn = nn.BatchNorm3d(current_channels)
        
        # Decoder layers with transposed convolutions
        self.decoder_layers = nn.ModuleList()
        self.decoder_bns = nn.ModuleList() if use_bn else None
        
        for i in range(num_layers - 2):
            next_channels = current_channels // 2
            self.decoder_layers.append(
                nn.ConvTranspose3d(current_channels, next_channels, 
                                  kernel_size=(3, 3, 3), stride=(1, 2, 2), 
                                  padding=(1, 1, 1), output_padding=(0, 1, 1))
            )
            if use_bn:
                self.decoder_bns.append(nn.BatchNorm3d(next_channels))
            current_channels = next_channels
        
        # Final layer to match output dimensions
        self.final_conv = nn.Conv3d(current_channels, out_channels, 
                                   kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # Residual connections for better gradient flow
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Keep temporal dim, pool spatial to 1x1
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 3D CNN
        
        Args:
            x: [B, C_in, T, H, W] input tensor (batch, channels, time, height, width)
        
        Returns:
            [B, C_out, H, W] predictions for the next time step
        """
        # Store original input for residual connection
        x_original = x
        
        # Encoder path
        encoder_features = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.use_bn:
                x = self.encoder_bns[i](x)
            x = F.relu(x)
            x = F.dropout3d(x, p=self.dropout, training=self.training)
            
            if i < len(self.encoder_layers) - 1:  # Don't store before bottleneck
                encoder_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        if self.use_bn:
            x = self.bottleneck_bn(x)
        x = F.relu(x)
        x = F.dropout3d(x, p=self.dropout, training=self.training)
        
        # Decoder path with skip connections
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if self.use_bn:
                x = self.decoder_bns[i](x)
            
            # Add skip connection if dimensions match
            if i < len(encoder_features):
                skip_idx = len(encoder_features) - 1 - i
                if x.shape[2:] == encoder_features[skip_idx].shape[2:]:  # Check T, H, W dimensions
                    x = x + encoder_features[skip_idx]
            
            x = F.relu(x)
            x = F.dropout3d(x, p=self.dropout, training=self.training)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Residual connection if input and output channels don't match
        if self.residual_conv is not None:
            residual = self.residual_conv(x_original)
            # Ensure temporal dimension matches by taking last time step
            if residual.size(2) > x.size(2):
                residual = residual[:, :, -x.size(2):, :, :]
            x = x + residual
        
        # The output should be [B, C_out, H, W] - take the last temporal prediction
        # Since we're predicting the next time step, we output the last temporal slice
        if x.dim() == 5 and x.size(2) > 1:
            x = x[:, :, -1, :, :]  # [B, C_out, H, W]
        
        return x


class SimpleCNN3D(nn.Module):
    """A simpler version of 3D CNN for faster training and less memory usage"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        # Build consecutive Conv3D layers
        for i in range(num_layers):
            next_channels = hidden_channels if i < num_layers - 1 else out_channels
            layers.append(
                nn.Conv3d(current_channels, next_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            )
            if use_bn and i < num_layers - 1:  # No BN on output layer
                layers.append(nn.BatchNorm3d(next_channels))
            if i < num_layers - 1:  # No ReLU on output layer
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout3d(p=dropout))
            
            current_channels = next_channels
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for simple 3D CNN
        
        Args:
            x: [B, C_in, T, H, W] input tensor
            
        Returns:
            [B, C_out, H, W] predictions for the next time step
        """
        x = self.layers(x)
        
        # Take the last temporal prediction
        if x.dim() == 5 and x.size(2) > 1:
            x = x[:, :, -1, :, :]  # [B, C_out, H, W]
        
        return x
