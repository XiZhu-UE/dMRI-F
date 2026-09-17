import torch
from torch import nn
from .blocks import ResnetBlocks


class EncodeBlock_3DUNet(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 num_res_blocks:int,
                 norm_type:str,
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        resnets = []
        for i in range(num_res_blocks):
            in_channels = self.in_channels if i == 0 else self.out_channels
            resnets.append(
                ResnetBlocks(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    norm_type=norm_type,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.downsampler = nn.MaxPool3d(kernel_size=2,stride=2)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        h = x
        for resnet in self.resnets:
            h = resnet(h)
        h = self.downsampler(h)
        return h


class DecodeBlock_3DUnet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_res_blocks: int,
                 norm_type: str,
                 upsample_type: str,
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.upsample_type = upsample_type
        resnet = []
        for i in range(self.num_res_blocks):
            in_channels = in_channels
            resnet.append(
                ResnetBlocks(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    norm_type=norm_type,
                )
            )
        self.resnet = nn.ModuleList(resnet)
        if self.upsample_type == 'deconv':
            self.upsampler = nn.ConvTranspose3d(in_channels=self.out_channels, out_channels=self.out_channels,
                                                kernel_size=2, stride=2)
        elif self.upsample_type == 'interpolate':
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels,
                          kernel_size=(1, 1, 1), stride=1)
            )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        h = x
        for resnet in self.resnet:
            h = resnet(x)
        h = self.upsampler(h)
        return h


class MidBlock_3DUnet(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 num_res_blocks:int,
                 norm_type: str,
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        resnets = []
        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlocks(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    norm_type=norm_type,
                )
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        h = x
        for resnet in self.resnets:
            h = resnet(h)
        return h


class Unet_3D(nn.Module):
    def __init__(self,
                 in_channel: int,
                 depth: int,
                 out_channel: int,
                 num_res_blocks: list,
                 num_channels: list,
                 norm_type: str,
                 upsample_type: str,
                 ) -> None:
        super().__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.num_res_blocks = num_res_blocks
        self.depth = depth
        self.num_channels = num_channels
        self.ini_conv = nn.Conv3d(in_channels=self.in_channels, out_channels=self.num_channels[0], stride=1,
                                  kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.upsample_type = upsample_type
        output_channel = num_channels[0]
        for i in range(1,depth):
            in_channel = output_channel
            output_channel = num_channels[i]
            down_block = EncodeBlock_3DUNet(
                in_channels=in_channel,
                out_channels=output_channel,
                num_res_blocks=num_res_blocks[i],
                norm_type=norm_type,
            )
            self.down_blocks.append(down_block)
        for i in range(2,depth):
            in_channel = self.num_channels[-i]*2
            out_channel = self.num_channels[-(i+1)]
            up_block = DecodeBlock_3DUnet(
                in_channels=in_channel,
                out_channels=out_channel,
                num_res_blocks=num_res_blocks[-i],
                norm_type=norm_type,
                upsample_type=self.upsample_type,
            )
            self.up_blocks.append(up_block)
        self.mid_block = MidBlock_3DUnet(in_channels=self.num_channels[self.depth-1],
                                         out_channels=self.num_channels[self.depth-2],
                                      num_res_blocks=self.num_res_blocks[self.depth-1],
                                         norm_type=norm_type)
        if self.upsample_type == 'deconv':
            self.upsampler = nn.ConvTranspose3d(in_channels=self.num_channels[self.depth - 2],
                                                out_channels=self.num_channels[self.depth - 2],
                                                kernel_size=2, stride=2)
        elif self.upsample_type == 'interpolate':
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_channels=self.num_channels[self.depth - 2], out_channels=self.num_channels[self.depth - 2],
                          kernel_size=(1, 1, 1), stride=1)
            )
        self.fin_conv_1 = ResnetBlocks(in_channels=self.num_channels[0]*2,
                                     out_channels=self.num_channels[0],
                                       norm_type=norm_type,)
        self.fin_conv_2 = nn.Conv3d(in_channels=self.num_channels[0],out_channels=self.out_channels,
                                    kernel_size=(1,1,1),
                                    stride=1,
                                    padding=0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ini_conv(x)
        skip_connections = []
        skip_connections.append(h)
        for down_block in self.down_blocks:
            h = down_block(h)
            skip_connections.append(h)

        h = self.mid_block(h)
        h = self.upsampler(h)

        for i, up_block in enumerate(self.up_blocks):
            h = torch.cat([h, skip_connections[-(i+2)]],dim=1)
            h = up_block(h)

        h = torch.cat([h, skip_connections[0]],dim=1)
        h = self.fin_conv_1(h)
        output = self.fin_conv_2(h)

        return output
