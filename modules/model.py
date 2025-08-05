import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init


class DataEncoder(nn.Module):
    """Time-series data encoder module."""

    def __init__(self, d_model, embed_dim, num_heads, d_meta_dim):
        """
        Args:
            d_model: Input feature dimension of the time-series data.
            embed_dim: Target embedding dimension.
            num_heads: Number of attention heads (not used here, kept for compatibility).
            d_meta_dim: Dimension of data meta-embedding (not used in forward, kept for API consistency).
        """
        super().__init__()
        # Linear layer to project input features to embedding space
        self.data_linear = nn.Linear(d_model, embed_dim)

    def forward(self, x, d_meta_embed):
        """
        Forward pass for data encoding.

        Args:
            x: Input time-series data [batch_size, num_patches, d_model]
            d_meta_embed: Data meta-embedding (unused in this version)

        Returns:
            x: Patch-level embeddings [batch_size, num_patches, embed_dim]
            mean_emb: Global average pooled embedding [batch_size, embed_dim]
        """
        x = self.data_linear(x)  # Project to embedding dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        mean_emb = x.mean(dim=1)  # Global average pooling over patches
        return x, mean_emb

    def functional_forward(self, x, d_meta_embed, params=None):
        """
        Functional forward pass supporting external parameters (e.g., for MAML-style updates).

        Args:
            x: Input time-series data.
            d_meta_embed: Data meta-embedding (unused).
            params: Optional dict of external parameters (fast weights).

        Returns:
            x: Patch-level embeddings.
            mean_emb: Pooled global embedding.
        """
        if params is None:
            x = self.data_linear(x)
        else:
            # Use external parameters for linear layer
            weight = params['data_encoder.data_linear.weight']
            bias = params.get('data_encoder.data_linear.bias', None)
            x = F.linear(x, weight, bias)

        if x.dim() == 2:
            x = x.unsqueeze(0)
        mean_emb = x.mean(dim=1)  # Global average pooling
        return x, mean_emb


class ModelEncoder(nn.Module):
    """Model encoder that combines meta, topology, and functional embeddings."""

    def __init__(self, meta_dim=23, meta_outdim=64, topo_dim=128, func_dim=96, embed_dim=512):
        """
        Args:
            meta_dim: Dimension of model meta-embedding (e.g., architecture stats).
            meta_outdim: Intermediate dimension for meta features.
            topo_dim: Dimension of topology embedding.
            func_dim: Dimension of functional embedding.
            embed_dim: Final output embedding dimension.
        """
        super().__init__()
        # Map high-level model metadata to a fixed intermediate dimension
        self.meta_mapping = nn.Sequential(
            nn.Linear(meta_dim, meta_outdim),
            nn.ReLU()
        )

        # Project concatenated features to final embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(meta_outdim + topo_dim + func_dim, embed_dim),
            nn.ReLU(),
            # LayerNorm can be added for stability, currently commented out
        )

    def forward(self, meta_embed, topo_embed, func_embed):
        """
        Forward pass for model encoding.

        Args:
            meta_embed: Model metadata embedding [batch_size, zoo_size, meta_dim]
            topo_embed: Topology embedding [batch_size, zoo_size, topo_dim]
            func_embed: Functional embedding [batch_size, zoo_size, func_dim]

        Returns:
            embedded: Combined model embedding [batch_size, zoo_size, embed_dim]
        """
        if meta_embed.dim() == 2:
            meta_embed = meta_embed.unsqueeze(0)  # Add batch dimension if missing
        bs = meta_embed.shape[0]

        # Transform metadata
        meta_embed = self.meta_mapping(meta_embed)

        # Concatenate all model-related embeddings
        combined = torch.cat([meta_embed, topo_embed, func_embed], dim=-1)

        # Project to final embedding space
        embedded = self.projection(combined)
        return embedded

    def functional_forward(self, meta_embed, topo_embed, func_embed, params=None):
        """
        Functional forward pass with support for external parameters.

        Args:
            meta_embed: Model metadata embedding.
            topo_embed: Topology embedding.
            func_embed: Functional embedding.
            params: External parameters (fast weights).

        Returns:
            embedded: Final model embedding.
        """
        if meta_embed.dim() == 2:
            meta_embed = meta_embed.unsqueeze(0)
        bs = meta_embed.shape[0]

        if params is None:
            meta_embed = self.meta_mapping(meta_embed)
            combined = torch.cat([meta_embed, topo_embed, func_embed], dim=-1)
            embedded = self.projection(combined)
        else:
            # Manually apply meta_mapping using external parameters
            meta_weight = params['model_encoder.meta_mapping.0.weight']
            meta_bias = params['model_encoder.meta_mapping.0.bias']
            meta_embed = F.linear(meta_embed, meta_weight, meta_bias)
            meta_embed = F.relu(meta_embed)

            # Concatenate features
            combined = torch.cat([meta_embed, topo_embed, func_embed], dim=-1)

            # Manually apply projection
            proj_weight = params['model_encoder.projection.0.weight']
            proj_bias = params['model_encoder.projection.0.bias']
            embedded = F.linear(combined, proj_weight, proj_bias)
            embedded = F.relu(embedded)
        return embedded


class MetaModel(nn.Module):
    """Meta-model that integrates data and model encoders with a Mixture-of-Experts (MoE) predictor."""

    def __init__(self, d_model, embed_dim, num_heads, meta_dim, meta_outdim, topo_dim, func_dim, d_meta_dim,
                 num_experts=4):
        """
        Args:
            d_model: Input dimension of time-series data.
            embed_dim: Embedding dimension for both data and model.
            num_heads: Number of attention heads (not used directly).
            meta_dim: Dimension of model metadata.
            meta_outdim: Intermediate dimension for model metadata.
            topo_dim: Topology embedding dimension.
            func_dim: Functional embedding dimension.
            d_meta_dim: Data meta-embedding dimension (unused in current implementation).
            num_experts: Number of experts in the MoE layer.
        """
        super().__init__()
        # Initialize encoders
        self.data_encoder = DataEncoder(d_model, embed_dim, num_heads, d_meta_dim)
        self.model_encoder = ModelEncoder(meta_dim, meta_outdim, topo_dim, func_dim, embed_dim)

        # MOE components
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_experts)  # Each expert predicts a scalar score
        ])
        self.gating = nn.Sequential(
            nn.Linear(1, num_experts),  # Gating network takes horizon as input
            nn.Softmax(dim=-1)  # Outputs normalized weights over experts
        )

    def forward(self, x, m_meta_emb, d_meta_emb, topo_emb, func_emb, horizon):
        """
        Standard forward pass.

        Args:
            x: Time-series input [batch_size, num_patches, d_model]
            m_meta_emb: Model metadata [batch_size, zoo_size, meta_dim]
            d_meta_emb: Data metadata (unused)
            topo_emb: Topology embeddings [batch_size, zoo_size, topo_dim]
            func_emb: Functional embeddings [batch_size, zoo_size, func_dim]
            horizon: Forecasting horizon (used for gating) [batch_size]

        Returns:
            prediction: Final prediction scores [batch_size, zoo_size]
            mean_embed: Data encoder's global embedding
            model_emb: Model embeddings before attention
            attn_output: Attention output
        """
        # Encode data and model
        data_emb, mean_embed = self.data_encoder(x, d_meta_emb)
        model_emb = self.model_encoder(m_meta_emb, topo_emb, func_emb)  # [bs, zoo_size, dim]

        # Cross-attention: model_emb queries data_emb
        attn_output, _ = self.cross_attention(query=model_emb, key=data_emb, value=data_emb)

        # Normalize horizon and prepare for gating
        horizon = horizon / 720
        horizon = horizon.unsqueeze(-1)

        # Gating network
        gate_weights = self.gating(horizon).unsqueeze(0)  # [1, num_experts] or [bs, num_experts]

        # Expert outputs
        expert_outputs = torch.stack([e(attn_output) for e in self.experts], dim=-1).squeeze(-2)
        # Shape: [batch_size, zoo_size, num_experts]

        # Weighted sum of expert outputs
        prediction = (expert_outputs * gate_weights.unsqueeze(1)).sum(dim=-1)  # [batch_size, zoo_size]

        return prediction, mean_embed, model_emb, attn_output

    def functional_forward(self, x, m_meta_emb, d_meta_emb, topo_emb, func_emb, horizon, params=None):
        """
        Functional forward pass supporting external parameters (e.g., for meta-learning).

        Args:
            Same as forward, plus:
            params: Dict of external parameters (fast weights).

        Returns:
            prediction, mean_embed, model_emb, attn_output
        """
        if params is None:
            params = dict(self.named_parameters())

        # Data encoder with external parameters
        data_emb, mean_embed = self.data_encoder.functional_forward(
            x, d_meta_emb,
            params={k: v for k, v in params.items() if 'data_encoder' in k}
        )

        # Model encoder with external parameters
        model_emb = self.model_encoder.functional_forward(
            m_meta_emb, topo_emb, func_emb,
            params={k: v for k, v in params.items() if 'model_encoder' in k}
        )

        # Cross-attention (manual implementation for functional compatibility)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                query=model_emb,
                key=data_emb,
                value=data_emb,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
        else:
            # Fallback: manual attention computation
            attn_scores = (model_emb @ data_emb.transpose(-2, -1)) / (data_emb.size(-1) ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = attn_weights @ data_emb

        # Normalize horizon
        horizon = horizon / 720
        horizon = horizon.unsqueeze(-1)

        # Gating network with external parameters
        gate_weight = params['gating.0.weight']
        gate_bias = params['gating.0.bias']
        gate_weights = F.linear(horizon, gate_weight, gate_bias)
        gate_weights = F.softmax(gate_weights, dim=-1)
        if gate_weights.dim() == 1:
            gate_weights = gate_weights.unsqueeze(0)  # Add batch dimension

        # Compute outputs from each expert using external parameters
        expert_outputs = []
        for i in range(self.num_experts):
            weight = params[f'experts.{i}.weight']
            bias = params[f'experts.{i}.bias']
            expert_outputs.append(F.linear(attn_output, weight, bias))
        expert_outputs = torch.stack(expert_outputs, dim=-1).squeeze(-2)
        # Shape: [batch_size, zoo_size, num_experts]

        # Final prediction: weighted sum
        prediction = (expert_outputs * gate_weights.unsqueeze(1)).sum(dim=-1)

        return prediction, mean_embed, model_emb, attn_output