import torch
import torch.nn as nn
from timm.models.layers import DropPath
import math

class Two_WayTransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()

        self.q = nn.Linear(c // 2, c // 2, bias=False)
        self.k = nn.Linear(c // 2, c // 2, bias=False)
        self.v = nn.Linear(c // 2, c // 2, bias=False)
        self.ma = nn.MultiheadAttention(c // 2, num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        self.actSplit = nn.GELU()
        self.ln = nn.LayerNorm(c//2)

    def forward(self, x):
        SplitC = x.size()[2] // 2

        x1, x2 = torch.split(x, [SplitC, SplitC], dim=2)

        "TWO-Way multi-head processing"
        x1 = self.ln(self.ma(self.q(x1), self.k(x1), self.v(x1))[0]) + x1
        x2 = self.ln(self.ma(self.q(x2), self.k(x2), self.v(x2))[0]) + x2

        x = torch.cat([x1, x2], dim=2)

        return self.fc2(self.fc1(x)) + x

from timm.models.layers import DropPath


#two-branch correlated frame alignment network TCFANet
class MultiLayer(nn.Module):
    def __init__(self, c1, c2, drop_path):
        #print(c1, c2)
        super().__init__()
        self.input_dim = c1
        self.output_dim = c2
        self.norm = nn.LayerNorm(c1)
        self.ST = Two_WayTransformerLayer(c1 // 2, 8)
        self.ST_Y = Two_WayTransformerLayer(c1, 8)
        self.proj = nn.Linear(c2, c2)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
 
    def forward(self, x, y):
        ############################原图特征处理###########################
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        x_norm = self.norm(x_flat)

        x1, x2 = torch.chunk(x_norm, 2, dim=2)
        x_ST1 = self.ST(x1) + self.skip_scale * x1
        x_ST2 = self.ST(x2) + self.skip_scale * x2

        x_ST = torch.cat([x_ST1, x_ST2], dim=2)

        x_ST = self.norm(x_ST)
        x_ST = self.proj(x_ST)

        out_x = x_ST.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        ##########################关联真处理#############################
        B, C = y.shape[:2]
        assert C == self.input_dim
        n_tokensy = y.shape[2:].numel()
        img_dimsy = y.shape[2:]
        y_flat = y.reshape(B, C, n_tokensy).transpose(-1, -2)

        y_norm = self.norm(y_flat)

        y_ST = self.ST_Y(y_norm) + self.skip_scale * y_norm

        #y_ST = self.norm(y_ST)
        y_ST = self.proj(y_ST)

        out_y = y_ST.transpose(-1, -2).reshape(B, self.output_dim, *img_dimsy)
        #print((out_x+out_y).shape)
        out_y = self.drop_path(out_y)

        return out_x+out_y