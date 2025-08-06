import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        return x


class MarketRegimeTransformer(nn.Module):
    def __init__(self, lookback_window=127, n_features=5, d_model=128, nhead=8, 
                 num_feature_layers=3, num_aggregate_layers=2, dim_feedforward=512, 
                 dropout=0.1, num_classes=3):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        self.feature_transformers = nn.ModuleList([
            FeatureTransformer(
                input_dim=1,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_feature_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(n_features)
        ])
        
        self.feature_projection = nn.Linear(n_features * d_model, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.aggregate_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_aggregate_layers)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.duration_regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        
        feature_outputs = []
        for i in range(self.n_features):
            feature_input = x[:, :, i:i+1]
            feature_output = self.feature_transformers[i](feature_input)
            feature_outputs.append(feature_output)
        
        combined_features = torch.cat(feature_outputs, dim=-1)
        combined_features = self.feature_projection(combined_features)
        
        aggregated = self.aggregate_transformer(combined_features)
        
        pooled = self.global_pool(aggregated.transpose(1, 2)).squeeze(-1)
        
        class_logits = self.classifier(pooled)
        duration = self.duration_regressor(pooled)
        
        return class_logits, duration


if __name__ == "__main__":
    model = MarketRegimeTransformer(
        lookback_window=127,
        n_features=5,
        d_model=128,
        nhead=8,
        num_feature_layers=3,
        num_aggregate_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=3
    )
    
    dummy_input = torch.randn(32, 127, 5)
    class_logits, duration = model(dummy_input)
    print(f"Model output shapes - Classes: {class_logits.shape}, Duration: {duration.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")