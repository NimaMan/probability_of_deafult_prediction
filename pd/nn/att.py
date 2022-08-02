
import torch 
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, hidden_dim=None):
        super(DotProductAttention, self).__init__()

    def forward(self, query: torch.Tensor, value: torch.Tensor, key=None):
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query, value):
        score = self.score_proj(torch.tanh(self.query_proj(query))).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = attn.unsqueeze(1) * value
        return context, attn
