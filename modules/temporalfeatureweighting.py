import torch
import torch.nn as nn
import torchvision
import math

class FDConv(nn.Module):
    def __init__(self, in_c, out_c, k, p, s, d):
        super(FDConv, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k,
                                            padding=p, stride=s, groups=math.gcd(in_c, out_c), dilation=d),
                                  nn.BatchNorm2d(out_c),
                                  nn.SiLU())

    def forward(self, x):
        return self.conv(x)


class TemporalEncoderLayer(nn.Module):
    """Temporal Encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU()):
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
    def with_pos_embed(self, tensor, pos=None):
        """Add position embeddings if given."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):

        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TemporalFeatureWeightingModule(nn.Module):
    def __init__(self, in_c, out_c,  batch):
        super(TemporalFeatureWeightingModule, self).__init__()

        self.batch = batch

        self.encoder = FineGrainedTransformerLayer(4096, 4)
    def forward(self, x):
        B, C, H, W = x.size()
        y = x.clone()

        x = x.view(1, B*C, -1)
        print(x.shape)
        x = self.encoder(x)
        x = x.view(B, C, H, W)
        return x


class FineGrainedTransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()

        self.q = nn.Linear(c // 4, c // 4, bias=False)
        self.k = nn.Linear(c // 4, c // 4, bias=False)
        self.v = nn.Linear(c // 4, c // 4, bias=False)
        self.ma = nn.MultiheadAttention(c // 4, num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        self.bnSplit = nn.BatchNorm1d(c // 4)
        self.actSplit = nn.GELU()
        self.bn = nn.BatchNorm1d(c)
        self.act = nn.GELU()

    def forward(self, x):
        SplitC = x.size()[2] // 4

        x1, x2, x3, x4 = torch.split(x, [SplitC, SplitC, SplitC, SplitC], dim=2)

        "Four-way multi-head processing"
        x1 = self.actSplit(self.bnSplit(self.ma(self.q(x1), self.k(x1), self.v(x1))[0].permute(0, 2, 1))).permute(0, 2, 1) + x1
        x2 = self.actSplit(self.bnSplit(self.ma(self.q(x2), self.k(x2), self.v(x2))[0].permute(0, 2, 1))).permute(0, 2, 1) + x2
        x3 = self.actSplit(self.bnSplit(self.ma(self.q(x3), self.k(x3), self.v(x3))[0].permute(0, 2, 1))).permute(0, 2, 1) + x3
        x4 = self.actSplit(self.bnSplit(self.ma(self.q(x4), self.k(x4), self.v(x4))[0].permute(0, 2, 1))).permute(0, 2, 1) + x4

        x = torch.cat([x1, x2, x3, x4], dim=2)

        return self.fc2(self.fc1(x)) + x



from thop import profile
from thop import clever_format

if __name__ == '__main__':
    model = FineGrainedTransformerLayer(24, 64, 4)
    input = torch.randn(4, 24, 64, 64)
    flops, params = profile(model, inputs=(input,))
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))
    flops, params = clever_format([flops, params], '%.3f')
    print(f"运算量：{flops}, 参数量：{params}")



