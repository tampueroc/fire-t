import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, img_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Convolutional layer to extract patches and project them into embeddings
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = self.proj(x)  # Shape: [batch_size, embed_dim, num_patches_h, num_patches_w]
        x = x.flatten(2)  # Flatten spatial dimensions: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # Rearrange to [batch_size, num_patches, embed_dim]
        return x


class TemporalPatchEmbedding(nn.Module):
    def __init__(self, seq_length, in_channels, embed_dim, patch_size, img_size):
        super(TemporalPatchEmbedding, self).__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size, img_size)
        self.num_patches = self.patch_embed.num_patches
        self.total_patches = seq_length * self.num_patches

        # Positional embeddings for the combined sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_patches, embed_dim))

    def forward(self, x):
        # x: [batch_size, seq_length, channels, height, width]
        batch_size, seq_length, channels, height, width = x.shape
        # Reshape to process all images in the sequence at once
        x = x.view(batch_size * seq_length, channels, height, width)
        # Apply PatchEmbedding to each image
        x = self.patch_embed(x)  # [batch_size * seq_length, num_patches, embed_dim]
        # Reshape to combine sequences
        x = x.view(batch_size, self.total_patches, self.embed_dim)  # [batch_size, total_patches, embed_dim]
        # Add positional encoding
        x = x + self.pos_embed
        return x


