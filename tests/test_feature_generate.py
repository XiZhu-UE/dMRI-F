"""The feature example produces registration-ready NIfTI data."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import nibabel as nib
import numpy as np
import torch
from torch import nn
from dimf.features import load_config, build_encoder, generate_features
from dimf.downstream.inference import load_feature
from dimf.defaults import MODEL
from dimf.models.unet import Unet_3D


class FeatureExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.affine = np.eye(4)
        self.affine[0, 3] = 12
        data = np.full((9, 10, 11, 7), 20, dtype=np.float32)
        data[..., 0] = 10
        data[0, 0, 0] = 0
        self.input = self.root / 'input.nii.gz'
        nib.save(nib.Nifti1Image(data, self.affine), self.input)

    def settings(self):
        settings = load_config()
        settings.update(input=str(self.input), output=str(self.root / 'feature.nii.gz'),
                        patch_size=[8, 8, 8], patch_overlap=[4, 4, 4], device='cpu')
        return settings

    def test_defaults_are_relative_and_six_directions(self):
        settings = load_config()
        self.assertEqual(settings['source_directions'], 6)
        self.assertTrue(Path(settings['checkpoint']).is_absolute())
        self.assertTrue(Path(settings['output']).is_absolute())

    def test_feature_output_is_registration_ready(self):
        class Probe(nn.Module):
            def forward(self, source):
                return source[:, :1].expand(-1, 64, -1, -1, -1)
        settings = self.settings()
        result = generate_features(settings, Probe(), torch.device('cpu'))
        self.assertEqual(result.shape, (9, 10, 11, 64))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result[0, 0, 0, 0], 0)
        self.assertEqual(result[5, 5, 5, 0], 1)
        image, data = load_feature(settings['output'])
        np.testing.assert_array_equal(image.affine, self.affine)
        np.testing.assert_array_equal(data, result)

    def test_input_count_and_checkpoint_channels(self):
        settings = self.settings()
        data = np.ones((9, 10, 11, 6), dtype=np.float32)
        nib.save(nib.Nifti1Image(data, self.affine), self.input)
        with self.assertRaisesRegex(ValueError, '7 volumes'):
            generate_features(settings, nn.Identity(), torch.device('cpu'))
        with patch('dimf.features.torch.load', return_value={
                'ImageEncoder.ini_conv.weight': torch.empty(64, 5, 3, 3, 3)}):
            with self.assertRaisesRegex(ValueError, '6 DWI'):
                build_encoder('unused.pth', 6)

    def test_pretraining_encoder_weights_load_strictly(self):
        original = Unet_3D(in_channel=6, out_channel=MODEL['latent_channels'], **MODEL['unet'])
        checkpoint = self.root / 'pretrained.pth'
        torch.save({f'ImageEncoder.{name}': value for name, value in original.state_dict().items()}, checkpoint)
        loaded = build_encoder(checkpoint, 6)
        self.assertEqual(set(original.state_dict()), set(loaded.state_dict()))
        with torch.inference_mode():
            self.assertEqual(loaded(torch.ones(1, 6, 8, 8, 8)).shape, (1, 64, 8, 8, 8))


if __name__ == '__main__':
    unittest.main()
