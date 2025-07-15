# src/models/configuration_olmo.py
from typing import Dict, Optional

from transformers import PretrainedConfig

# Configuration constants with clear documentation
# Based on GPT-NeoX-20B tokenizer which is commonly used for financial text
FINPILE_VOCAB_SIZE = 50277  # Exact vocabulary size for FinPile tokenizer compatibility

# Standard architecture ratios following transformer best practices
INTERMEDIATE_SIZE_RATIO = 4  # Standard MLP expansion ratio (hidden_size * 4)
HEAD_DIM = 64  # Standard attention head dimension for most transformer models

# Model scaling parameters based on empirical research
MODEL_SCALING_CONFIG = {
    # Format: (hidden_size, num_layers, num_heads, target_params_millions)
    "4M": (64, 8, 8, 4),
    "6M": (96, 8, 8, 6),
    "8M": (128, 8, 8, 8),
    "10M": (144, 8, 8, 10),
    "14M": (192, 8, 8, 14),
    "16M": (208, 8, 8, 16),
    "20M": (192, 8, 8, 20),  # Different layer config for efficiency
    "60M": (384, 16, 12, 60),  # Start using more layers
    "90M": (528, 16, 12, 90),
    "150M": (768, 12, 12, 150),  # Standard transformer-base dimensions
    "300M": (1024, 16, 16, 300),
    "530M": (1344, 16, 16, 530),
    "750M": (1536, 16, 16, 750),
    "1B": (2048, 16, 16, 1000),  # 1 billion parameters
}


class OLMoConfig(PretrainedConfig):
    """Configuration class for OLMo models."""

    model_type = "olmo"

    def __init__(
        self,
        vocab_size: int = 50277,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swiglu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_cache: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        use_bias: bool = False,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_bias = use_bias
        self.tie_word_embeddings = tie_word_embeddings


class ModelConfigFactory:
    """Factory for creating OLMo configurations with validation and consistency checks."""

    @staticmethod
    def create_config(model_size: str, **overrides) -> OLMoConfig:
        """Create a model configuration for the specified size.

        Args:
            model_size: Model size identifier (e.g., "4M", "150M", "1B")
            **overrides: Additional parameters to override defaults

        Returns:
            OLMoConfig: Configured model

        Raises:
            ValueError: If model_size is not supported or configuration is invalid
        """
        if model_size not in MODEL_SCALING_CONFIG:
            available_sizes = list(MODEL_SCALING_CONFIG.keys())
            raise ValueError(f"Unsupported model size '{model_size}'. Available: {available_sizes}")

        hidden_size, num_layers, num_heads, target_params = MODEL_SCALING_CONFIG[model_size]

        # Calculate intermediate size with standard ratio
        intermediate_size = hidden_size * INTERMEDIATE_SIZE_RATIO

        # Validate head dimensions
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})")

        head_dim = hidden_size // num_heads
        # DataDecide uses head dimensions from 8 to 64 based on empirical validation
        if head_dim < 4 or head_dim > 128:
            raise ValueError(
                f"Head dimension {head_dim} is outside valid range [4, 128]. DataDecide uses head_dim >= 8."
            )

        config = OLMoConfig(
            vocab_size=FINPILE_VOCAB_SIZE,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            **overrides,  # Allow custom overrides
        )

        return config

    @staticmethod
    def get_available_sizes() -> list[str]:
        """Get list of available model sizes."""
        return list(MODEL_SCALING_CONFIG.keys())

    @staticmethod
    def estimate_parameters(model_size: str) -> int:
        """Estimate the number of parameters for a model size."""
        if model_size not in MODEL_SCALING_CONFIG:
            raise ValueError(f"Unknown model size: {model_size}")
        return MODEL_SCALING_CONFIG[model_size][3] * 1_000_000  # Convert millions to actual count


# Predefined configurations using the factory pattern
def _create_legacy_configs() -> Dict[str, OLMoConfig]:
    """Create legacy configuration dictionary for backward compatibility."""
    configs = {}
    for size in MODEL_SCALING_CONFIG.keys():
        configs[size] = ModelConfigFactory.create_config(size)
    return configs


# Legacy configuration dictionary - maintained for backward compatibility
OLMO_CONFIGS = _create_legacy_configs()
