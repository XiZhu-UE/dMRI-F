"""Active registration architecture and losses from the supplied VoxelMorph code."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class U_Network_feat_all(nn.Module):

    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(U_Network_feat_all, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 128 if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 4, 2, batchnorm=bn))
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn))
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn))
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))
        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + 128, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        nd = Normal(0, 1e-05)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.batch_norm = getattr(nn, 'BatchNorm{0}d'.format(dim))(3)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, 'Conv{0}d'.format(dim))
        bn_fn = getattr(nn, 'BatchNorm{0}d'.format(dim))
        if batchnorm:
            layer = nn.Sequential(conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding), bn_fn(out_channels), nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding), nn.LeakyReLU(0.2))
        return layer

    def forward(self, src, tgt):
        src_norm = src
        tgt_norm = tgt
        x = torch.cat([src_norm, tgt_norm], dim=1)
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        y = self.dec[3](y)
        y = self.dec[4](y)
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        if self.vm2:
            y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow

class IntegrateVelocityField(nn.Module):

    def __init__(self, shape: tuple, steps: int=1, interpolation_mode: str='bilinear', align_corners: bool=False, device: str='cpu'):
        super().__init__()
        if steps < 0:
            raise ValueError(f'steps should be >= 0, found: {steps}')
        self.steps = steps
        self.scale = 1.0 / 2 ** self.steps
        self.transformer = SpatialTransformer(shape, interpolation_mode)
        self.transformer.to(device)

    def forward(self, velocity_field: torch.Tensor) -> torch.Tensor:
        velocity_field = velocity_field * self.scale
        for _ in range(self.steps):
            velocity_field = velocity_field + self.transformer(velocity_field, velocity_field)
        return velocity_field

class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False)

def ncc_loss(I, J, filt_channel, device, win=None):
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], 'volumes should be 1 to 3 dimensions. found: %d' % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, filt_channel, *win]).to(device)
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-05)
    return -1 * torch.mean(cc)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = (I * I, J * J, I * J)
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return (I_var, J_var, cross)



def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])
    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0
