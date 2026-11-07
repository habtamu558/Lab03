
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Divide each 28x28 image into 16 patches.
        self.patch_embedding = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=7
        )

        # Create two Transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Classify using the mean of all patch representations.
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # Convert image into patch embeddings.
        x = self.patch_embedding(x)

        # Change shape to (batch, patches, features).
        x = x.flatten(2).transpose(1, 2)

        # Process patches using attention.
        x = self.transformer(x)

        # Mean pooling across all patches.
        x = x.mean(dim=1)

        # Produce 10 class scores.
        x = self.classifier(x)

        return x
