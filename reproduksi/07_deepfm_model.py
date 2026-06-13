"""
07_deepfm_model.py
====================
DeepFM model implementation in PyTorch.

Architecture:
- FM Component: First-order + Second-order (Factorization Machine)
- DNN Component: MLP with ReLU activation and Dropout
- Linear Component: Sum of bias terms

Reference:
- Guo et al. (2017) "DeepFM: A Factorization-Machine based Neural Network"

Usage:
    from07_deepfm_model import DeepFM
    model = DeepFM(field_config)

Output:
    This module is imported by 08_deepfm_training.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# DeepFM Model
# ============================================================================

class DeepFM(nn.Module):
    """
    DeepFM Model for Recommender System.

    Combines:
    - FM Component: Captures 2nd-order feature interactions via factorization
    - DNN Component: Captures higher-order feature interactions via MLP
    - Linear Component: First-order feature importance

    Args:
        field_config: dict, Configuration with feature dimensions
        embed_dim: int, Embedding dimension (default: 16)
        dnn_hidden_dims: list, Hidden layer dimensions for DNN
        dropout: float, Dropout rate (default: 0.5)
        use_sentiment: bool, Whether to include sentiment features
    """

    def __init__(
        self,
        field_config,
        embed_dim=16,
        dnn_hidden_dims=[400, 400, 400],
        dropout=0.5,
        use_sentiment=True
    ):
        super(DeepFM, self).__init__()

        self.field_config = field_config
        self.embed_dim = embed_dim
        self.use_sentiment = use_sentiment

        # =========================================================================
        # Embedding Layer
        # =========================================================================
        self.embeddings = nn.ModuleDict()

        # Sparse features (need embedding)
        sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip_code']

        for feat_name in sparse_features:
            num_embeddings = field_config['feature_dims'][feat_name]
            self.embeddings[feat_name] = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embed_dim
            )

        # =========================================================================
        # Linear Component (First-order)
        # =========================================================================
        self.linear_embeddings = nn.ModuleDict()

        for feat_name in sparse_features:
            num_embeddings = field_config['feature_dims'][feat_name]
            self.linear_embeddings[feat_name] = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=1
            )

        # Genres linear (multi-hot, use linear layer)
        self.genres_linear = nn.Linear(
            field_config['feature_dims']['genres'],
1,
            bias=True
        )

        # Sentiment linear (dense feature)
        if use_sentiment:
            self.sentiment_linear = nn.Linear(1, 1, bias=True)

        # =========================================================================
        # DNN Component
        # =========================================================================
        # Calculate input dimension for DNN
        # All sparse features (6) + genres (18) =24 features
        # Each feature has embed_dim embedding
        num_sparse_features = len(sparse_features)  # 6
        num_genre_features = field_config['feature_dims']['genres']  # 18

        dnn_input_dim = num_sparse_features * embed_dim + num_genre_features

        if use_sentiment:
            dnn_input_dim += 1  # Add sentiment feature

        self.dnn = nn.ModuleList()
        prev_dim = dnn_input_dim

        for hidden_dim in dnn_hidden_dims:
            self.dnn.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.dnn_dropout = nn.Dropout(dropout)
        self.dnn_output = nn.Linear(prev_dim, 1)

        # =========================================================================
        # Initialize weights
        # =========================================================================
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sparse_features, dense_features):
        """
        Forward pass.

        Args:
            sparse_features: dict, Sparse feature tensors
                - user_id: (batch_size,)
                - movie_id: (batch_size,)
                - gender: (batch_size,)
                - age: (batch_size,)
                - occupation: (batch_size,)
                - zip_code: (batch_size,)
                - genres: (batch_size, num_genres)

            dense_features: dict, Dense feature tensors
                - sentiment: (batch_size, 1) or None

        Returns:
            torch.Tensor: Predictions (batch_size, 1) with sigmoid applied
        """
        batch_size = sparse_features['user_id'].size(0)

        # =========================================================================
        # Linear Component (First-order)
        # =========================================================================
        linear_output = 0.0

        sparse_feat_names = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip_code']

        for feat_name in sparse_feat_names:
            linear_output += self.linear_embeddings[feat_name](sparse_features[feat_name]).squeeze(-1)

        # Genres linear
        linear_output += self.genres_linear(sparse_features['genres']).squeeze(-1)

        # Sentiment linear
        if self.use_sentiment and dense_features is not None and 'sentiment' in dense_features:
            linear_output += self.sentiment_linear(dense_features['sentiment']).squeeze(-1)

        # =========================================================================
        # FM Component (Second-order)
        # =========================================================================
        # Get embeddings for all sparse features
        embed_list = []
        for feat_name in sparse_feat_names:
            embed = self.embeddings[feat_name](sparse_features[feat_name])
            embed_list.append(embed)

        # Concatenate embeddings: (batch_size, num_features, embed_dim)
        embed_matrix = torch.stack(embed_list, dim=1)

        # First-order: sum of embeddings
        first_order = embed_matrix.sum(dim=1)  # (batch_size, embed_dim)

        # Second-order: 0.5 * (sum^2 - sum(squares))
        sum_square = (embed_matrix.sum(dim=1)) ** 2  # (batch_size, embed_dim)
        square_sum = (embed_matrix ** 2).sum(dim=1)  # (batch_size, embed_dim)
        second_order = 0.5 * (sum_square - square_sum)  # (batch_size, embed_dim)

        # FM output
        fm_output = first_order + second_order  # (batch_size, embed_dim)

        # =========================================================================
        # DNN Component
        # =========================================================================
        # Flatten embeddings
        embed_flat = embed_matrix.view(batch_size, -1)  # (batch_size, num_features * embed_dim)

        # Add genres (already flat from input)
        dnn_input = torch.cat([embed_flat, sparse_features['genres']], dim=1)

        # Add sentiment if available
        if self.use_sentiment and dense_features is not None and 'sentiment' in dense_features:
            dnn_input = torch.cat([dnn_input, dense_features['sentiment']], dim=1)

        # Pass through DNN
        for layer in self.dnn:
            dnn_input = F.relu(layer(dnn_input))
            dnn_input = self.dnn_dropout(dnn_input)

        dnn_output = self.dnn_output(dnn_input).squeeze(-1)  # (batch_size,)

        # =========================================================================
        # Combine All Components
        # =========================================================================
        output = linear_output + fm_output.sum(dim=1) + dnn_output

        # Apply sigmoid
        output = torch.sigmoid(output)

        return output


# ============================================================================
# DataLoader for DeepFM
# ============================================================================

class DeepFMDataset(torch.utils.data.Dataset):
    """Dataset for DeepFM training."""

    def __init__(self, data_dict, use_sentiment=True):
        self.data = data_dict
        self.use_sentiment = use_sentiment

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):
        sparse_features = {
            'user_id': torch.tensor(self.data['user_id'][idx], dtype=torch.long),
            'movie_id': torch.tensor(self.data['movie_id'][idx], dtype=torch.long),
            'gender': torch.tensor(self.data['gender'][idx], dtype=torch.long),
            'age': torch.tensor(self.data['age'][idx], dtype=torch.long),
            'occupation': torch.tensor(self.data['occupation'][idx], dtype=torch.long),
            'zip_code': torch.tensor(self.data['zip_code'][idx], dtype=torch.long),
            'genres': torch.tensor(self.data['genres'][idx], dtype=torch.float32)
        }

        dense_features = None
        if self.use_sentiment:
            dense_features = {
                'sentiment': torch.tensor([self.data['sentiment'][idx]], dtype=torch.float32)
            }

        label = torch.tensor(self.data['labels'][idx], dtype=torch.float32)

        return sparse_features, dense_features, label


def collate_fn(batch):
    """Custom collate function for batching."""
    sparse_features_list = []
    dense_features_list = []
    labels_list = []

    for sparse, dense, label in batch:
        sparse_features_list.append(sparse)
        dense_features_list.append(dense)
        labels_list.append(label)

    # Stack sparse features
    stacked_sparse = {}
    for key in sparse_features_list[0].keys():
        stacked_sparse[key] = torch.stack([s[key] for s in sparse_features_list]).squeeze(-1)

    # Stack dense features
    stacked_dense = None
    if dense_features_list[0] is not None:
        stacked_dense = {}
        for key in dense_features_list[0].keys():
            stacked_dense[key] = torch.stack([d[key] for d in dense_features_list])

    # Stack labels
    labels = torch.stack(labels_list)

    return stacked_sparse, stacked_dense, labels


# ============================================================================
# Test Model
# ============================================================================

if __name__ == '__main__':
    # Test model
    field_config = {
        'feature_dims': {
            'user_id': 6040,
            'movie_id': 3952,
            'gender': 2,
            'age': 7,
            'occupation': 21,
            'zip_code': 3439,
            'genres': 18,
            'sentiment': 1
        }
    }

    model = DeepFM(field_config, embed_dim=16, use_sentiment=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    sparse_features = {
        'user_id': torch.randint(0, 6040, (batch_size,)),
        'movie_id': torch.randint(0, 3952, (batch_size,)),
        'gender': torch.randint(0, 2, (batch_size,)),
        'age': torch.randint(0, 7, (batch_size,)),
        'occupation': torch.randint(0, 21, (batch_size,)),
        'zip_code': torch.randint(0, 3439, (batch_size,)),
        'genres': torch.randint(0, 2, (batch_size, 18)).float()
    }

    dense_features = {
        'sentiment': torch.rand(batch_size, 1)
    }

    output = model(sparse_features, dense_features)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
