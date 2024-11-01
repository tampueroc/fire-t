import torch
import torch.nn as nn
from src.models.embedding_handler import TemporalPatchEmbedding


class FireTransformer(nn.Module):
    def __init__(self, seq_length, in_channels, embed_dim, num_heads, num_layers, patch_size, img_size, num_classes):
        super().__init__()
        self.temporal_patch_embed = TemporalPatchEmbedding(seq_length, in_channels, embed_dim, patch_size, img_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_length, channels, height, width]
        x = self.temporal_patch_embed(x)  # [batch_size, total_patches, embed_dim]
        x = x.transpose(0, 1)  # [total_patches, batch_size, embed_dim]
        x = self.transformer_encoder(x)  # [total_patches, batch_size, embed_dim]
        x = x.mean(dim=0)  # Aggregate over patches
        x = self.classifier(x)
        return x


class FireTransformerWithWeather(nn.Module):
    def __init__(self, seq_length, in_channels, embed_dim, num_heads, num_layers, patch_size, img_size, num_classes, weather_dim):
        super().__init__()
        self.temporal_patch_embed = TemporalPatchEmbedding(seq_length, in_channels, embed_dim, patch_size, img_size)
        self.weather_embed = nn.Linear(weather_dim, embed_dim)
        self.num_patches = self.temporal_patch_embed.num_patches
        self.total_tokens = seq_length * self.num_patches + seq_length  # Patches + weather tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_tokens, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, weather):
        # x: [batch_size, seq_length, channels, height, width]
        # weather: [batch_size, seq_length, weather_dim]
        batch_size = x.shape[0]
        patch_embeddings = self.temporal_patch_embed(x)  # [batch_size, total_patches, embed_dim]

        # Embed weather data
        weather_embeddings = self.weather_embed(weather)  # [batch_size, seq_length, embed_dim]
        weather_embeddings = weather_embeddings.view(batch_size, -1, self.embed_dim)  # Flatten weather embeddings

        # Combine patch and weather embeddings
        embeddings = torch.cat((patch_embeddings, weather_embeddings), dim=1)  # [batch_size, total_tokens, embed_dim]
        embeddings = embeddings + self.pos_embed
        embeddings = embeddings.transpose(0, 1)  # [total_tokens, batch_size, embed_dim]

        x = self.transformer_encoder(embeddings)
        x = x.mean(dim=0)  # Aggregate over tokens
        x = self.classifier(x)
        return x


