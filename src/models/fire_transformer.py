import torch.nn as nn
from models.embedding_handler import TemporalPatchEmbedding


class FiresTransformer(nn.Module):
    def __init__(self, seq_length, in_channels, embed_dim, num_heads, num_layers, patch_size, img_size):
        super(FireTransformer, self).__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size

        # Temporal patch embedding layer
        self.temporal_patch_embed = TemporalPatchEmbedding(
            seq_length=seq_length,
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.decoder = nn.Linear(embed_dim, self.temporal_patch_embed.num_patches)

    def forward(self, x):
        # x: [batch_size, seq_length, channels, height, width]
        batch_size = x.size(0)

        # Generate temporal patch embeddings
        x = self.temporal_patch_embed(x)  # [batch_size, total_patches, embed_dim]

        # Transpose for transformer input
        x = x.transpose(0, 1)  # [total_patches, batch_size, embed_dim]

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # [total_patches, batch_size, embed_dim]

        # Transpose back
        x = x.transpose(0, 1)  # [batch_size, total_patches, embed_dim]

        # Use only the last time step's patches for prediction
        # Assuming the last set of patches corresponds to the latest time step
        last_time_step_patches = x[:, -self.temporal_patch_embed.num_patches:, :]  # [batch_size, num_patches, embed_dim]

        # Aggregate embeddings (e.g., via mean)
        x = last_time_step_patches.mean(dim=1)  # [batch_size, embed_dim]

        # Output layer
        output = self.decoder(x)  # [batch_size, num_patches]

        # Reshape output to image format
        patch_dim = int(self.temporal_patch_embed.num_patches ** 0.5)
        output = output.view(batch_size, 1, patch_dim, patch_dim)  # [batch_size, 1, patch_dim, patch_dim]

        # Upsample to original image size
        output = nn.functional.interpolate(output, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)  # [batch_size, 1, img_size, img_size]

        return output

