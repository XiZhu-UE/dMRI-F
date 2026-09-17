import torch
from torch import nn


class ResnetBlocks(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 norm_type: str,
                 norm_num_groups: int = 16,
                 norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if norm_type == 'GN':
            self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps,
                                      affine=True)
        elif norm_type == 'IN':
            self.norm1 = nn.InstanceNorm3d(num_features=in_channels, affine=True, eps=norm_eps)
            self.norm2 = nn.InstanceNorm3d(num_features=out_channels, affine=True, eps=norm_eps)
        else:
            self.norm1 = None
            self.norm2 = None
        self.nonlinearity = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(3,3,3), stride=1,
                               padding=1)
        self.conv2 = nn.Conv3d(in_channels=self.out_channels,out_channels=self.out_channels, kernel_size=(3,3,3),stride=1,
                               padding=1)
        if self.in_channels == self.out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels,
                                             kernel_size=(1, 1, 1), stride=1, padding=0)


    def forward(self, x:torch.Tensor)->torch.Tensor:
        h = x
        if self.norm1 is not None:
            h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        if self.norm2 is not None:
            h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        return self.skip_connection(x)+h
