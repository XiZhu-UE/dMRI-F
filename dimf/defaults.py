"""Stable architecture and implementation defaults for the validated model."""

MODEL = {'depth': 8,
 'width': 256,
 'skips': [4],
 'latent_channels': 64,
 'feature_channels': 128,
 'decoder_channels': 16,
 'unet': {'depth': 3,
          'num_res_blocks': [1, 1, 1],
          'num_channels': [64, 128, 256],
          'norm_type': 'IN',
          'upsample_type': 'interpolate'}}

ENCODING = {'position_frequencies': 10, 'spherical_harmonic_levels': 4}

GEOMETRY = {'patch_size': [64, 64, 64], 'pad_before': [3, 1, 3], 'pad_after': [4, 1, 4]}

TRAINING = {'resume_if_exists': True,
 'visualization_pattern': 'epoch_diff00{epoch}.png',
 'visualization_slice': 32,
 'visualization_channels': [0, 10, 20, 30],
 'samples_per_subject': 20,
 'random_downsample_probability': 0.5,
 'direction_flip_probability': 0.3,
 'downsample_spacing': 2,
 'upsample_spacing': 1,
 'perceptual_network': 'alex',
 'loader': {'batch_size': 1,
            'num_workers': 8,
            'prefetch_factor': 2,
            'persistent_workers': True,
            'pin_memory': True},
 'train_shuffle': True,
 'validation_shuffle': True}
