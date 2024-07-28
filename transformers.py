import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(e)
        dot = F.softmax(dot, dim=2)
        assert not contains_nan(dot[:, 1:, :])
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, num_classes, mask=False):
        super().__init__()
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=k, heads=heads, mask=mask))
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        x = self.tblocks(x.unsqueeze(1))  # Add a dummy dimension to match the expected input shape
        return x


embedding_dim = 64
num_heads = 8
num_layers = 6
num_classes = 10
mask = False

# Sample input tensor
batch_size = 1
sample_input = torch.randn(batch_size, embedding_dim)

# Initialize model
model = Transformer(k=embedding_dim, heads=num_heads, depth=num_layers, num_classes=num_classes, mask=mask)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Example inputs
X_12 = torch.randn(12, embedding_dim)
X_k = torch.randn(8, embedding_dim)
x = torch.randn(batch_size, embedding_dim)  # Example input tensor
T = 0.5
class_index = 4

# def contrastive_loss(X_12, X_k, x, T, class_index):
#     Z_12, Z_k = [], []

#     # Compute Z_12
#     for i in range(len(X_12)):
#         Z_12.append(model(X_12[i].unsqueeze(0)))  # Ensure input shape matches (batch_size, embedding_dim)

#     # Compute Z_k
#     for i in range(len(X_k)):
#         Z_k.append(model(X_k[i].unsqueeze(0)))  # Ensure input shape matches (batch_size, embedding_dim)

#     # Compute z and z_class
#     z = model(x)
#     z_class = Z_k[class_index]

#     # Compute denominator
#     denominator = 0
#     for i in range(len(X_12)):
#         denominator += torch.exp(nn.functional.cosine_similarity(z, Z_12[i], dim=1) / T)
#     for i in range(len(X_k)):
#         denominator += torch.exp(nn.functional.cosine_similarity(z, Z_k[i], dim=1) / T)

#     # Compute loss
#     numerator = torch.exp(nn.functional.cosine_similarity(z, z_class, dim=1) / T)
#     L = -torch.log(numerator / denominator)

#     return L

def contrastive_loss(X_12, X_k, x, T, class_index):
    Z_12, Z_k = [], []

    # Compute Z_12
    for i in range(len(X_12)):
        Z_12.append(model(X_12[i].unsqueeze(0)))  # Ensure input shape matches (batch_size, embedding_dim)

    # Compute Z_k
    for i in range(len(X_k)):
        Z_k.append(model(X_k[i].unsqueeze(0)))  # Ensure input shape matches (batch_size, embedding_dim)

    # Compute z and z_class
    z = model(x)
    z_class = Z_k[class_index]

    # Compute denominator
    denominator = 0
    for i in range(len(X_12)):
        denominator += torch.exp(nn.functional.cosine_similarity(z, Z_12[i], dim=1) / T)
    for i in range(len(X_k)):
        denominator += torch.exp(nn.functional.cosine_similarity(z, Z_k[i], dim=1) / T)

    # Compute loss
    numerator = torch.exp(nn.functional.cosine_similarity(z, z_class, dim=1) / T)
    L = -torch.log(numerator / denominator)

    return L.mean()  # Reduce to scalar


num_epochs = 10
train_features = torch.randn(1000, embedding_dim)
train_features = train_features.to(device)

# Create DataLoader
train_dataset = TensorDataset(train_features)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0

    for batch_features in train_loader:
        batch_features = batch_features[0].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Compute loss
        loss = contrastive_loss(X_12, X_k, batch_features, T, class_index)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
