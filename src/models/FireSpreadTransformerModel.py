import torch
import torch.nn as nn
import einops

class FireSpreadTransformer(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, num_heads=8, num_layers=6, num_features=10, sequence_length=3):
        super(FireSpreadTransformer, self).__init__()

        # Patch embedding for each input frame
        self.patch_embed = nn.Conv2d(in_channels=1 + num_features, out_channels=embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # Positional encoding for sequence length
        self.positional_encoding = nn.Parameter(torch.randn(sequence_length, embed_dim))

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

        # MLP for weather input integration
        self.weather_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Final prediction layer (upscaling)
        self.prediction_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Binary mask output
        )

    def forward(self, fire_frames, weather):
        # Patch embedding and temporal positional encoding
        patches = self.patch_embed(fire_frames)  # Shape: [B, C, H', W']
        patches = einops.rearrange(patches, 'b c h w -> b (h w) c')

        # Add temporal positional encoding
        patches += self.positional_encoding[:patches.size(1)]

        # Flatten patches for transformer input
        patches = einops.rearrange(patches, 'b n e -> n b e')

        # Transformer encoding
        encoded = self.transformer_encoder(patches)

        # Aggregate output for prediction (e.g., mean over sequence dimension)
        encoded_mean = encoded.mean(dim=0)

        # Integrate weather information
        weather_embed = self.weather_mlp(weather).unsqueeze(0)
        encoded_mean = encoded_mean + weather_embed

        # Upsample and predict isochrone mask
        isochrone_pred = einops.rearrange(encoded_mean, 'b (h w) e -> b e h w', h=patches.size(1)**0.5, w=patches.size(1)**0.5)
        isochrone_mask = self.prediction_head(isochrone_pred)

        return isochrone_mask

