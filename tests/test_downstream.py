"""Small CPU checks for downstream data, objectives and checkpoint structure."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import nibabel as nib
import numpy as np
import torch
from torch import nn
from dimf.downstream.config import load_config
from dimf.downstream.data import PatchDataset, RegistrationDataset
from dimf.downstream.models import DownstreamModel
from dimf.downstream.registration import U_Network_feat_all
from dimf.downstream.training import Objective, run_training


class DownstreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.shape = (18, 20, 18)
        for subject in ('sub_01', 'sub_02'):
            mask = np.ones(self.shape, dtype=np.float32)
            b0 = np.full(self.shape, 10, dtype=np.float32)
            labels = np.full(self.shape, 2, dtype=np.float32)
            labels[0] = 77
            for suffix, data in [('mask', mask), ('b0_mean', b0), ('DKT_parc', labels),
                                 ('T1w_masked', b0 * 2),
                                 ('feature', np.ones((*self.shape, 64), dtype=np.float32))]:
                nib.save(nib.Nifti1Image(data, np.eye(4)), self.root / f'{subject}_{suffix}.nii.gz')
            (self.root / f'{subject}_xmax.json').write_text(json.dumps({'1000': 20}))
            for i in range(3):
                np.save(self.root / f'{subject}_bval_1000.0_bvec_1.000000_0.000000_0.000000_vol_{i:04d}.npy', b0)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, task):
        config = load_config(task)
        for key in list(config):
            if key.endswith('_dir'):
                config[key] = str(self.root)
        config.update(train_slice=[0, 1], validation_slice=[1, 2], device='cpu',
                      epochs=1, num_workers=0, samples_per_subject=1,
                      checkpoint=str(self.root / f'{task}.pth'), resume_if_exists=False)
        if task == 'reg':
            config.update(volume_size=[16, 16, 16], mask_erosion=0)
        else:
            config.update(source_directions=2, patch_size=[16, 16, 16])
        if task == 'syn':
            config['perceptual_weight'] = 0
        return config

    def test_relative_config_paths(self):
        path = Path(load_config('parc')['npy_dir'])
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.parts[-3:], ('data', 'prepared', 'volumes'))
        for task in ('parc', 'syn', 'reg'):
            config = load_config(task)
            self.assertTrue(Path(config['checkpoint']).is_absolute())

    def test_patch_data_formats_and_normalization(self):
        for task in ('parc', 'syn'):
            dataset = PatchDataset(self.config(task))
            sample = dataset[0]
            self.assertEqual(sample['source'].shape, (2, 16, 16, 16))
            self.assertEqual(dataset.subjects, ['sub_01'])
            valid = sample['mask'] > 0
            np.testing.assert_allclose(sample['source'][:, valid], 0.5)
            np.testing.assert_allclose(sample['b0'][valid], 1)
            if task == 'parc':
                self.assertEqual(set(np.unique(dataset.records['sub_01']['target'])), {0, 1})
            else:
                np.testing.assert_allclose(sample['target'][valid], 1)

    def test_registration_crop_and_subject_ids(self):
        config = self.config('reg')
        sample = RegistrationDataset(config)[0]
        self.assertEqual(sample['moving_features'].shape, (64, 16, 16, 16))
        np.testing.assert_allclose(sample['moving_b0'], 1)

    def test_checkpoint_keys_and_forward_shapes(self):
        for task in ('parc', 'syn'):
            config = load_config(task)
            with torch.device('meta'):
                model = DownstreamModel(config, load_pretrained=False)
                keys = model.state_dict()
                self.assertTrue(any(k.startswith('pretrain_model.') for k in keys))
                self.assertTrue(any(k.startswith('encoder.') for k in keys))
                self.assertTrue(any(k.startswith('decoder.') for k in keys))
                result = model(torch.empty(1, 6, 16, 16, 16), torch.empty(1, 16, 16, 16))
                self.assertEqual(result.shape, (1, 98 if task == 'parc' else 1, 16, 16, 16))
                self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))

    def test_three_training_loops_save_reload(self):
        class SmallPatchModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.decoder = nn.Conv3d(3, config['num_classes'] if config['task'] == 'parc' else 1, 1)

            def forward(self, source, b0):
                return self.decoder(torch.cat((source, b0.unsqueeze(1)), dim=1))

        def small_registration(*args):
            return U_Network_feat_all(3, [4, 8, 8, 8], [8, 8, 8, 8, 8, 4, 4])

        for task in ('parc', 'syn', 'reg'):
            config = self.config(task)
            with patch('dimf.downstream.training.DownstreamModel', SmallPatchModel), \
                 patch('dimf.downstream.training.U_Network_feat_all', small_registration):
                history = run_training(config)
                self.assertTrue(np.isfinite(history[0]['train_loss']))
                self.assertTrue(Path(config['checkpoint']).is_file())
                self.assertTrue((self.root / 'loss_curve.png').is_file())
                model = small_registration() if task == 'reg' else SmallPatchModel(config)
                model.load_state_dict(torch.load(config['checkpoint'], weights_only=True))

    def test_synthesis_perceptual_term_receives_gradients(self):
        config = self.config('syn')
        objective = Objective(config, 'cpu')
        config['perceptual_weight'] = 0.01
        objective.perceptual = nn.MSELoss()
        class ConstantModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.value = nn.Parameter(torch.tensor(0.25))

            def forward(self, source, b0):
                return self.value.expand(source.shape[0], 1, *source.shape[-3:])
        model = ConstantModel()
        batch = {key: torch.as_tensor(value).unsqueeze(0)
                 for key, value in PatchDataset(config)[0].items()}
        loss, metrics, _ = objective(model, batch)
        loss.backward()
        self.assertGreater(metrics['perceptual'], 0)
        self.assertTrue(torch.isfinite(model.value.grad))


if __name__ == '__main__':
    unittest.main()
