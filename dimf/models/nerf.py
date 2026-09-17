import torch
from torch import nn
import torch.nn.functional as F
from .unet import Unet_3D

class NeRF(nn.Module):
    def __init__(self, image_input_channel, D=8, W=256, input_ch=63, input_ch_views=16, skips=(4,), latent_channel=64, feature_dim=128, decoder_channels=16, unet=None):
        super(NeRF, self).__init__()
        self.image_input_channel = image_input_channel
        self.D = D
        self.latent_channel = latent_channel
        self.feature_dim = feature_dim
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.decoder_channels = decoder_channels
        unet = dict(unet or dict(depth=3, num_res_blocks=[1,1,1], num_channels=[64,128,256], norm_type='IN', upsample_type='interpolate'))
        self.ImageEncoder = Unet_3D(in_channel=image_input_channel, out_channel=latent_channel, **unet)
        self.Final_Decoder = Unet_3D(in_channel=decoder_channels, out_channel=1, **unet)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch+self.feature_dim, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
             for i in range(D - 1)]
        )

        self.feature_linear = nn.Linear(self.latent_channel+self.input_ch_views, self.feature_dim)
        self.conv1d = nn.Conv1d(
            in_channels=self.image_input_channel,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.output_linear = nn.Sequential(
            nn.Linear(W + input_ch_views, W // 2),
            nn.ReLU(),
            nn.Linear(W // 2, decoder_channels)
        )

    def forward(self, target_feature, source_patch, source_bvec_feature, image_size,  valid_coords_local):
        image_feature_upsampled = self.ImageEncoder(source_patch)

        coords = valid_coords_local.long()
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

        voxel_feature = image_feature_upsampled[0, :, x, y, z]
        voxel_feature = voxel_feature.permute(1, 0).unsqueeze(1)

        voxel_feature_expanded = voxel_feature.expand(-1, self.image_input_channel, -1)
        fused_feature = torch.cat([voxel_feature_expanded, source_bvec_feature], dim=-1)
        fused_feature = self.feature_linear(fused_feature)

        input_pts, input_views = torch.split(target_feature, [self.input_ch, self.input_ch_views], dim=-1)
        input_pts = input_pts.float()
        input_views = input_views.float()
        input_pts_expended = input_pts.expand(-1, self.image_input_channel, -1)
        h = torch.cat([input_pts_expended, fused_feature], dim=-1)
        h = self.conv1d(h)
        h = F.silu(h)
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        h = torch.cat([h, input_views], dim=-1)
        output = self.output_linear(h).squeeze(1)
        spatial_shape = tuple(source_patch.shape[-3:])
        pred_patch = output.new_zeros(self.decoder_channels, *spatial_shape)
        linear_idx = coords[:, 0] * spatial_shape[1] * spatial_shape[2] + coords[:, 1] * spatial_shape[2] + coords[:, 2]
        pred_patch_flat = pred_patch.view(self.decoder_channels, -1)
        pred_patch_flat.scatter_(1, linear_idx.unsqueeze(0).expand(self.decoder_channels, -1), output.T)
        pred_patch = pred_patch_flat.view(1, self.decoder_channels, *spatial_shape)
        pred_patch = self.Final_Decoder(pred_patch)

        return pred_patch, image_feature_upsampled





def build_model(config, source_directions):
    model = config['model']
    encoding = config['encoding']
    return NeRF(
        image_input_channel=source_directions, D=model['depth'], W=model['width'],
        input_ch=3 + 6 * encoding['position_frequencies'],
        input_ch_views=encoding['spherical_harmonic_levels'] ** 2,
        skips=model['skips'], latent_channel=model['latent_channels'],
        feature_dim=model['feature_channels'], decoder_channels=model['decoder_channels'],
        unet=model['unet'],
    )
