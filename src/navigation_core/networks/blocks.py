from torch import nn


class BlockAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BlockAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)


class BlockResidualSmallLayerNorm(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(BlockResidualSmallLayerNorm, self).__init__()
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


class BlockResidualSmall(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(BlockResidualSmall, self).__init__()
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


class BlockResidual(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(BlockResidual, self).__init__()
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


class LayerLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(LayerLinear, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        return self.block(x)


class LayerLeakyBatchNorm(nn.Module):
    def __init__(self, input_size, output_size):
        super(LayerLeakyBatchNorm, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


class LayerLeaky(nn.Module):
    def __init__(self, input_size, output_size):
        super(LayerLeaky, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)
