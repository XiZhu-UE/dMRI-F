"""Reuse the validated pretraining encoder and preserve checkpoint key names."""
import torch
from torch import nn
from ..defaults import MODEL, ENCODING
from ..models.nerf import build_model
from ..models.unet import Unet_3D


class DownstreamModel(nn.Module):
    def __init__(self, config, load_pretrained=True):
        super().__init__()
        self.pretrain_model = build_model({'model': MODEL, 'encoding': ENCODING},
                                          config['source_directions'])
        if load_pretrained:
            self.pretrain_model.load_state_dict(torch.load(
                config['pretrain_checkpoint'], map_location='cpu', weights_only=True))
        self.encoder = self.pretrain_model.ImageEncoder
        parc = config['task'] == 'parc'
        self.decoder = Unet_3D(
            in_channel=65, depth=3 if parc else 4,
            out_channel=config['num_classes'] if parc else 1,
            num_res_blocks=[2, 2, 2] if parc else [1, 1, 1, 1],
            num_channels=[128, 256, 512] if parc else [64, 128, 256, 512],
            norm_type='IN', upsample_type='interpolate')
        if config['freeze_encoder']:
            self.encoder.requires_grad_(False)

    def forward(self, source, b0):
        features = self.encoder(source)
        return self.decoder(torch.cat([features, b0.unsqueeze(1)], dim=1))
