import torch.nn as nn
import torch
class CSTL(nn.Module):
    def __init__(self, num_classes=125, cnn_channels=[32, 64, 128, 256], mste_scales=[1, 2, 4, 8],
                 attention_heads=8, dropout=0.3):
        super(CSTL, self).__init__()

        # CNN Backbone
        self.cnn_backbone = nn.Sequential(
            # First block
            nn.Conv2d(1, cnn_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels[0], momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),

            # Second block
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels[1], momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),

            # Third block
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels[2], momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),

            # Fourth block
            nn.Conv2d(cnn_channels[2], cnn_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels[3], momentum=0.1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Multi-Scale Temporal Encoding (MSTE)
        self.mste = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(cnn_channels[3], cnn_channels[3] // 2, kernel_size=scale, stride=scale),
                nn.BatchNorm1d(cnn_channels[3] // 2, momentum=0.1),
                nn.ReLU(inplace=True)
            ) for scale in mste_scales
        ])

        # Adaptive Temporal Attention (ATA)
        self.attention = nn.MultiheadAttention(
            embed_dim=cnn_channels[3] // 2,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels[3] // 2, 512),
            nn.BatchNorm1d(512, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # Ensure we have at least 2 samples for BatchNorm
        if batch_size == 1:
            # Duplicate the input to create a batch of 2
            x = torch.cat([x, x], dim=0)
            batch_size = 2

        # Reshape for CNN processing
        x = x.view(-1, 1, 64, 64)  # [batch_size * seq_len, 1, 64, 64]

        # CNN backbone
        x = self.cnn_backbone(x)  # [batch_size * seq_len, channels, 1, 1]
        x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, channels]

        # Multi-Scale Temporal Encoding
        mste_outputs = []
        for mste_layer in self.mste:
            # Process each temporal scale
            scale_output = mste_layer(x.transpose(1, 2))  # [batch_size, channels, seq_len]
            mste_outputs.append(scale_output.transpose(1, 2))  # [batch_size, seq_len, channels]

        # Concatenate multi-scale features
        x = torch.cat(mste_outputs, dim=1)  # [batch_size, seq_len * num_scales, channels]

        # Adaptive Temporal Attention
        attn_output, _ = self.attention(x, x, x)  # [batch_size, seq_len * num_scales, channels]

        # Global temporal pooling
        x = attn_output.mean(dim=1)  # [batch_size, channels]

        # Classification
        x = self.classifier(x)  # [batch_size, num_classes]

        # If we duplicated the input, return only the first sample
        if batch_size == 2 and x.size(0) == 2:
            x = x[0:1]

        return x