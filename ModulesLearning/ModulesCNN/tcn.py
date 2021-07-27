import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.modules.activation import MultiheadAttention

activation_dict = {"ReLU": torch.nn.ReLU(), "Softplus": torch.nn.Softplus(), "Softmax": torch.nn.Softmax}

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, attention=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]


            #if (attention == True) and (i%2 == 0):
             #   layers += [ConvAttentionBlock(num_channels[i])]


        if attention == True:
            layers += [ConvspatialAttentionBlock(num_channels[-1])]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AttentionBlock(nn.Module):

#Should this be followed by a positional embedding??
  """An attention mechanism similar to Vaswani et al (2017)
  The input of the AttentionBlock is `BxTxD` where `B` is the input
  minibatch size, `T` is the length of the sequence `D` is the dimensions of
  each feature.
  The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
  attention values.
  Arguments:
      dims (int): the number of dimensions (or channels) of each element in
          the input sequence
      k_size (int): the size of the attention keys
      v_size (int): the size of the attention values
      seq_len (int): the length of the input and output sequences
  """
  def __init__(self, dims, k_size, v_size, seq_len=None):
    super(AttentionBlock, self).__init__()
    self.key_layer = nn.Linear(dims, k_size)
    self.query_layer = nn.Linear(dims, k_size)
    self.value_layer = nn.Linear(dims, v_size)
    # self.query_layer = nn.Conv1d(in_channels=dims, out_channels=k_size//8, kernel_size=1)
    # self.key_layer = nn.Conv1d(in_channels=dims, out_channels=k_size//8, kernel_size=1)
    # self.value_layer = nn.Conv1d(in_channels=dims, out_channels=k_size, kernel_size=1)
    self.sqrt_k = math.sqrt(k_size)

  def forward(self, minibatch):

    minibatch = minibatch.permute(0,2,1)
    keys = self.key_layer(minibatch)
    queries = self.query_layer(minibatch)
    values = self.value_layer(minibatch)
    logits = torch.bmm(queries, keys.transpose(2,1))
    # Use numpy triu because you can't do 3D triu with PyTorch
    # TODO: using float32 here might break for non FloatTensor inputs.
    # Should update this later to use numpy/PyTorch types of the input.
    mask = np.triu(np.ones(logits.size()), k=1).astype('bool')
    mask = torch.from_numpy(mask)
    if torch.cuda.is_available():
        mask = mask.cuda()
    # do masked_fill_ on data rather than Variable because PyTorch doesn't
    # support masked_fill_ w/-inf directly on Variables for some reason.
    logits.data.masked_fill_(mask, float('-inf'))
    probs = F.softmax(logits, dim=1) / self.sqrt_k
    read = torch.bmm(probs, values)
    return (minibatch + read).permute(0,2,1)



class ConvspatialAttentionBlock(nn.Module):
  """
    Similar to the SAGAN paper: https://discuss.pytorch.org/t/attention-in-image-classification/80147/3
  """
# I haven't used any positional encoding here, not sure if needs to be there
  def __init__(self, dims):
    super(ConvspatialAttentionBlock, self).__init__()
    self.query_layer = nn.Conv1d(in_channels=dims, out_channels=dims//8, kernel_size=1)
    self.key_layer = nn.Conv1d(in_channels=dims, out_channels=dims//8, kernel_size=1)
    self.value_layer = nn.Conv1d(in_channels=dims, out_channels=dims, kernel_size=1)

    self.softmax = nn.Softmax(dim=-1)
    self.gamma = nn.Parameter(torch.zeros(1))

  def forward(self, minibatch):
    keys = self.key_layer(minibatch)
    queries = self.query_layer(minibatch).permute(0,2,1)
    values = self.value_layer(minibatch)
    logits = torch.bmm(queries, keys)
    mask = np.triu(np.ones(logits.size()), k=1).astype('bool')
    mask = torch.from_numpy(mask)
    #if torch.cuda.is_available():
     #   mask = mask.cuda()
    # do masked_fill_ on data rather than Variable because PyTorch doesn't
    # support masked_fill_ w/-inf directly on Variables for some reason.
    #logits.data.masked_fill_(mask, float('-inf'))

    probs = self.softmax(logits)
    read = torch.bmm(values, probs.permute(0,2,1))
    return (self.gamma*read)+minibatch


class ConvchannelAttentionBlock(nn.Module):
  """
    Similar to the SAGAN paper: https://discuss.pytorch.org/t/attention-in-image-classification/80147/3
  """
# I haven't used any positional encoding here, not sure if needs to be there
  def __init__(self, dims):
    super(ConvchannelAttentionBlock, self).__init__()
    # self.query_layer = nn.Conv1d(in_channels=dims, out_channels=dims//8, kernel_size=1)
    # self.key_layer = nn.Conv1d(in_channels=dims, out_channels=dims//8, kernel_size=1)
    # self.value_layer = nn.Conv1d(in_channels=dims, out_channels=dims, kernel_size=1)

    self.softmax = nn.Softmax(dim=-1)
    self.eta = nn.Parameter(torch.zeros(1))

  def forward(self, minibatch):
    keys = minibatch.permute(0,2,1)
    queries = minibatch
    values = minibatch
    logits = torch.bmm(queries, keys)
    mask = np.triu(np.ones(logits.size()), k=1).astype('bool')
    mask = torch.from_numpy(mask)
    #if torch.cuda.is_available():
     #   mask = mask.cuda()
    # do masked_fill_ on data rather than Variable because PyTorch doesn't
    # support masked_fill_ w/-inf directly on Variables for some reason.
    #logits.data.masked_fill_(mask, float('-inf'))
    logits_new = torch.max(logits, -1, keepdim=True)[0].expand_as(logits)-logits
    probs = self.softmax(logits_new)
    read = torch.bmm(probs,values)
    return (self.eta*read)+minibatch


class ConvAttentionBlock(nn.Module):
  """
    Similar to the SAGAN paper: https://discuss.pytorch.org/t/attention-in-image-classification/80147/3
  """
# I haven't used any positional encoding here, not sure if needs to be there
  def __init__(self, dims):
    super(ConvAttentionBlock, self).__init__()
    self.spatial_attention = ConvspatialAttentionBlock(dims)
    self.channel_attention = ConvchannelAttentionBlock(dims)
    #self.conv1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv1d(dims, dims, 1))
    #self.conv2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv1d(dims, dims, 1))
    self.conv3 = nn.Sequential(nn.Dropout1d(0.1, False), nn.Conv1d(dims, dims, 1))
  def forward(self, minibatch):

    s = self.spatial_attention(minibatch)
    c = self.channel_attention(minibatch)
    
    return self.conv3(s+c)



class SimplePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a basic positional encoding"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class MultiAttnHeadSimple(torch.nn.Module):
    """A simple multi-head attention model inspired by Vaswani et al."""

    def __init__(
            self,
            d_model=128,
            num_heads=2,
            dropout=0.5):
        super().__init__()
        self.pe = SimplePositionalEncoding(d_model)
        self.multi_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout)


    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:

        # x = self.dense_shape(x)

        # Permute to (L, B, M)
        #print("input shape:", x.shape)
        x = x.permute(2,0,1)

        x = self.pe(x)
        #print("After positional encoding shape:", x.shape)

        self.mask = mask

        if self.mask is None:
            device = x.device
            self.mask = self._generate_square_subsequent_mask(len(x)).to(device)

        # if mask is None:
        #     x = self.multi_attn(x, x, x)[0]
        # else:
        x = self.multi_attn(x, x, x, attn_mask=self.mask)[0]

        #print("Attention output shape:", x.shape)
        x = x.permute(1, 2, 0)

        #print("final return shape",x.shape)
        return x

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class ConvAttentionBlockv2(nn.Module):
    """
      Similar to the Dual Attention Network for Scene Segmentation paper: https://github.com/junfu1115/DANet/blob/master/encoding/models/sseg/danet.py
    """

    # I haven't used any positional encoding here, not sure if needs to be there
    def __init__(self, in_channels):
        super(ConvAttentionBlockv2, self).__init__()

        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(weight_norm(nn.Conv1d(in_channels, inter_channels, 2, padding=1, bias=False)),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(weight_norm(nn.Conv1d(in_channels, inter_channels, 2, padding=1, bias=False)),
                                    nn.ReLU())

        self.sa = ConvspatialAttentionBlock(inter_channels)
        self.sc = ConvchannelAttentionBlock(inter_channels)
        self.conv51 = nn.Sequential(weight_norm(nn.Conv1d(inter_channels, inter_channels, 2, padding=1, bias=False)),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(weight_norm(nn.Conv1d(inter_channels, inter_channels, 2, padding=1, bias=False)),
                                    nn.ReLU())

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout(0.1), nn.Conv1d(inter_channels, in_channels, 1))


    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output
