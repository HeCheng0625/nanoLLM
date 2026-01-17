import torch
import torch.nn as nn
import torch.nn.functional as F
from ..configuration_nanollm import NanoLLMConfig


class NanoMLP(nn.Module):
    def __init__(self, config: NanoLLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # TODO: Define 3 Linear layers (gate_proj, up_proj, down_proj)
        # Note: Qwen typically does NOT use bias in MLP linear layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # TODO: Define activation function (e.g., F.silu)
        self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch, seq_len, hidden_size]
        Output: [batch, seq_len, hidden_size]
        """
        # TODO: Implement SwiGLU: down(act(gate(x)) * up(x))
        output = self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return output
