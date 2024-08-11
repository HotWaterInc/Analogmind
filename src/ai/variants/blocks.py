from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)


class ResidualBlockSmallLayerNorm(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlockSmallLayerNorm, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlockSmallBatchNormWithAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlockSmallBatchNormWithAttention, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            AttentionLayer(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlockSmallBatchNorm(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlockSmallBatchNorm, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)


def _make_layer_linear(in_features, out_features):
    layer = nn.Sequential(
        nn.Linear(in_features, out_features),
        # nn.BatchNorm1d(out_features),
        # nn.LeakyReLU(),
    )
    return layer


def _make_layer(in_features, out_features):
    layer = nn.Sequential(
        nn.Linear(in_features, out_features),
        # nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )
    return layer
