import torch
from torch import nn


class GNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(GNN, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size

        self._input_layer = nn.Linear(self._input_size * 2, self._hidden_size)
        self._forget_gate = nn.Linear(self._input_size * 2, self._hidden_size)
        self._input_gate = nn.Linear(self._input_size * 2, self._hidden_size)

    def forward(self, input: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        :param input: Sentence representation. Tensor [1, input_size]
        :param h: hidden state of the previous GNN cell. Tensor [1, hidden_size]
        :return: Output of the GNN cell (= new hidden state). Tensor [1, hidden_size]
        """
        comp = torch.cat((input, h), dim=1)
        i_t = torch.sigmoid(self._input_layer(comp))
        f_t = torch.sigmoid(self._forget_gate(comp))
        g_t = torch.tanh(self._input_gate(comp))
        h_t = torch.tanh(i_t * g_t + f_t * h)  # "*" is element-wise multiplication here
        return h_t
