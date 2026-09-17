"""Check the new single-file prediction inputs and registration outputs."""
from pathlib import Path
import tempfile
import unittest
import nibabel as nib
import numpy as np
import torch
from torch import nn
from dimf.downstream.inference_config import load_inference_config
from dimf.downstream.inference import (load_patch_input, predict_patch_task,
                                       predict_registration, run_inference)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.affine = np.diag([2., 2., 2., 1.])
        self.affine[:3, 3] = [7, 8, 9]
        self.shape = (18, 20, 18)

    def nifti(self, name, data):
        path = self.root / name
        nib.save(nib.Nifti1Image(data.astype(np.float32), self.affine), path)
        return path

    def patch_config(self, task):
        config = load_inference_config(task)
        data = np.full((*self.shape, 7), 20, dtype=np.float32)
        data[..., 0] = 10
        data[0, 0, 0] = 0
        config.update(input=str(self.nifti('input.nii.gz', data)),
                      output=str(self.root / f'{task}.nii.gz'),
                      patch_size=[16, 16, 16], patch_overlap=[8, 8, 8], device='cpu')
        return config

    def test_default_six_directions_and_paths(self):
        for task in ('parc', 'syn'):
            config = load_inference_config(task)
            self.assertEqual(config['source_directions'], 6)
            self.assertTrue(Path(config['input']).is_absolute())
            self.assertTrue(Path(config['output']).is_absolute())
        self.assertEqual(load_inference_config('reg')['volume_size'], [144, 160, 144])

    def test_patch_prediction_original_space_and_input_order(self):
        for task in ('parc', 'syn'):
            config = self.patch_config(task)
            class Probe(nn.Module):
                def forward(self, source, b0):
                    self_shape = (source.shape[0], 98 if task == 'parc' else 1, *source.shape[-3:])
                    output = source.new_zeros(self_shape)
                    if task == 'parc':
                        output[:, 2] = 10 * source[:, 0]
                    else:
                        output[:, 0] = 0.5 * b0
                    return output
            result = predict_patch_task(config, Probe(), torch.device('cpu'))
            saved = nib.load(config['output'])
            self.assertEqual(result.shape, self.shape)
            np.testing.assert_array_equal(saved.affine, self.affine)
            self.assertEqual(result[0, 0, 0], 0)
            self.assertEqual(result[8, 9, 8], 2 if task == 'parc' else 0.5)
            if task == 'parc':
                self.assertEqual(saved.get_data_dtype(), np.dtype('int16'))

    def test_wrong_patch_volume_count(self):
        self.patch_config('syn')
        self.nifti('bad.nii.gz', np.ones((*self.shape, 6)))
        with self.assertRaisesRegex(ValueError, '7 volumes'):
            load_patch_input(str(self.root / 'bad.nii.gz'), 6, [16] * 3)

    def test_registration_flow_and_warped_b0(self):
        config = load_inference_config('reg')
        feature = np.ones((*self.shape, 64), dtype=np.float32)
        config.update(moving_feature=str(self.nifti('moving.nii.gz', feature)),
                      fixed_feature=str(self.nifti('fixed.nii.gz', feature)),
                      moving_b0=str(self.nifti('b0.nii.gz', np.full(self.shape, 10))),
                      flow_output=str(self.root / 'flow.nii.gz'),
                      warped_b0_output=str(self.root / 'warped.nii.gz'),
                      volume_size=[16, 16, 16], mask_erosion=0, device='cpu')
        class ZeroFlow(nn.Module):
            def forward(self, moving, fixed):
                self_shape = (1, 3, *moving.shape[-3:])
                return moving.new_zeros(self_shape)
        flow, warped = predict_registration(config, ZeroFlow(), torch.device('cpu'))
        self.assertEqual(flow.shape, (*self.shape, 3))
        self.assertEqual(warped.shape, self.shape)
        self.assertGreater(warped.max(), 0)
        np.testing.assert_array_equal(flow, 0)
        np.testing.assert_array_equal(warped[0], 0)
        np.testing.assert_array_equal(nib.load(config['flow_output']).affine, self.affine)
        np.testing.assert_array_equal(nib.load(config['warped_b0_output']).affine, self.affine)

    def test_checkpoint_source_count_checked_before_model_allocation(self):
        config = self.patch_config('parc')
        config['checkpoint'] = str(self.root / 'wrong.pth')
        torch.save({'encoder.ini_conv.weight': torch.empty(64, 5, 3, 3, 3)}, config['checkpoint'])
        with self.assertRaisesRegex(ValueError, 'accept 6 source DWI'):
            run_inference(config)


if __name__ == '__main__':
    unittest.main()
