"""Temporary synthetic images verify raw-data preparation and target-gradient inference."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import nibabel as nib
import torch
import yaml
from dimf.config import ROOT, load_config
from dimf.data.common import discover_subjects, load_gradients, padding_for_shape
from dimf.data.conversion import run_nii2npy
from dimf.data.shells import run_shell_max, load_xmax
from dimf.data.training import Generalized_NeRF_Dataset
from dimf.data.inference import load_input
from dimf.inference import predict_targets, save_prediction, check_source_count, encode_directions, run_inference
from dimf.models.nerf import build_model


def options(mode):
    return yaml.safe_load((ROOT / f'{mode}.ymal').read_text(encoding='utf-8'))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, mode, updates):
        values = options(mode)
        values.update(updates)
        path = self.root / f'{mode}.ymal'
        path.write_text(yaml.safe_dump(values), encoding='utf-8')
        return load_config(mode, path)

    def raw_subject(self, subject='sub-A', shape=(8, 8, 8), nested=True):
        folder = self.root / 'raw' / subject if nested else self.root / 'raw'
        folder.mkdir(parents=True, exist_ok=True)
        data = np.ones((*shape, 5), dtype=np.float64)
        data *= np.array([100, 10, 20, 30, 40])
        bvals = np.array([0, 1000, 1000, 2000, 3000])
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nib.save(nib.Nifti1Image(data, np.eye(4)), folder/f'{subject}_denoised.nii.gz')
        nib.save(nib.Nifti1Image(np.ones(shape, np.uint8), np.eye(4)), folder/f'{subject}_mask.nii.gz')
        np.savetxt(folder/f'{subject}_bval.bval', bvals)
        np.savetxt(folder/f'{subject}_bvecs.bvec', bvecs.T)
        return folder, data

    def dataset_config(self, **extra):
        return self.config('dataset', dict(input_dir=str(self.root/'raw'), output_dir=str(self.root/'npy'),
                                           metadata_dir=str(self.root/'metadata'), workers=1, **extra))

    def inference_config(self):
        b0 = np.full((8, 8, 8), 123, np.float32)
        source = np.stack([np.full_like(b0, 10), np.full_like(b0, 20)], -1)
        nib.save(nib.Nifti1Image(b0, np.eye(4)), self.root/'b0.nii.gz')
        nib.save(nib.Nifti1Image(source, np.eye(4)), self.root/'source.nii.gz')
        np.savetxt(self.root/'source.bval', [1000, 1000])
        np.savetxt(self.root/'source.bvec', [[1, 0], [0, 1], [0, 0]])
        np.savetxt(self.root/'target.bval', [2500, 1000, 2000])
        np.savetxt(self.root/'target.bvec', [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        config = self.config('inference', dict(
            b0='b0.nii.gz', source_dwi='source.nii.gz', source_bvals='source.bval', source_bvecs='source.bvec',
            target_bvals='target.bval', target_bvecs='target.bvec', checkpoint='weights.pth',
            output='prediction.nii.gz', patch_size=[8]*3, patch_overlap=[4]*3, num_workers=0,
            source_directions=2, device='cpu'))
        config['geometry'].update(pad_before=[0]*3, pad_after=[0]*3)
        return config

    def test_bulk_preparation_no_xmax_and_repeated_directions(self):
        _, data = self.raw_subject()
        self.raw_subject('sub-B', shape=(9, 10, 11), nested=False)
        config = self.dataset_config()
        self.assertEqual(run_nii2npy(config), [4, 4])
        files = sorted((self.root/'npy').glob('sub-A*.npy'))
        self.assertEqual(len(files), 4)
        self.assertEqual(np.load(files[0]).dtype, np.float64)
        np.testing.assert_array_equal(np.load(files[0]), data[..., 1])
        np.testing.assert_array_equal(np.load(files[1]), data[..., 2])
        self.assertEqual(load_xmax(self.root/'metadata/sub-A_xmax.json')[1000], 20)
        train = self.config('pretraining', dict(data_root='npy', metadata_root='metadata',
                            train_subjects=1, validation_subjects=1, source_directions=2, patch_size=[8]*3))
        train['training']['random_downsample_probability'] = 0
        dataset = Generalized_NeRF_Dataset(2, train['training']['data_root'], config=train)
        batch = dataset[0]
        self.assertEqual(batch['source_patch'].shape, (2, 8, 8, 8))
        self.assertEqual(batch['target_patch'].shape, (1, 8, 8, 8))
        validation = Generalized_NeRF_Dataset(2, train['training']['data_root'], is_Train=False, config=train)
        self.assertEqual(validation.subids, ['sub-B'])
        self.assertNotEqual(tuple(dataset.subject_dict['sub-A']['image_size']), tuple(validation.subject_dict['sub-B']['image_size']))

    def test_shell_preparation_includes_last_volumes_and_optional_mask(self):
        self.raw_subject()
        config = self.dataset_config(mask_pattern=None)
        run_shell_max(config)
        self.assertEqual(load_xmax(self.root/'metadata/sub-A_xmax.json'), {1000: 20, 2000: 30, 3000: 40})
        self.assertTrue((self.root/'metadata/sub-A_mask.nii.gz').exists())

    def test_missing_gradient_and_duplicate_ids_fail_early(self):
        folder, _ = self.raw_subject()
        (folder/'sub-A_bval.bval').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'missing bval'):
            run_nii2npy(self.dataset_config())
        self.raw_subject()
        self.raw_subject(nested=False)
        with self.assertRaisesRegex(ValueError, 'Duplicate subject ID'):
            discover_subjects(self.dataset_config())

    def test_gradient_count_and_single_direction(self):
        np.savetxt(self.root/'x.bval', [1000])
        np.savetxt(self.root/'x.bvec', [[1], [0], [0]])
        bvals, bvecs = load_gradients(self.root/'x.bval', self.root/'x.bvec', 1)
        self.assertEqual(bvecs.shape, (1, 3))
        self.assertEqual(bvals.shape, (1,))
        with self.assertRaises(ValueError):
            load_gradients(self.root/'x.bval', self.root/'x.bvec', 2)

    def test_hcp_padding_preserved(self):
        config = load_config('pretraining')
        self.assertEqual(padding_for_shape((145, 174, 145), config['geometry']), ((3, 4), (1, 1), (3, 4)))

    def test_target_order_b0_and_gradient_outputs(self):
        config = self.inference_config()
        inputs = load_input(config)
        class TargetProbe(torch.nn.Module):
            def forward(self, target, source, source_sh, size, coords):
                value = target[0, 0, -16:].sum().item()
                return source.new_full((1, 1, *source.shape[-3:]), value), source
        result = predict_targets(inputs, TargetProbe(), torch.device('cpu'), config)
        self.assertEqual(result.shape, (8, 8, 8, 4))
        np.testing.assert_array_equal(result[..., 0], inputs.b0)
        expected = encode_directions(inputs.target_bvals, inputs.target_bvecs, config).sum(-1).numpy() * inputs.scale
        for index in range(3):
            np.testing.assert_allclose(result[..., index+1], expected[index], rtol=1e-6)
        save_prediction(result, inputs, config['inference']['output'])
        output = nib.load(self.root/'prediction.nii.gz')
        np.testing.assert_array_equal(output.affine, inputs.image.affine)
        np.testing.assert_array_equal(np.loadtxt(self.root/'prediction.bval'), [0, 2500, 1000, 2000])
        np.testing.assert_array_equal(np.loadtxt(self.root/'prediction.bvec')[:, 0], [0, 0, 0])
        np.testing.assert_array_equal(np.loadtxt(self.root/'prediction.bvec')[:, 1:], inputs.target_bvecs.T)

    def test_parallel_preparation_and_custom_directory_layout(self):
        for subject in ('person_01', 'person_02'):
            folder, _ = self.raw_subject(subject)
            for suffix, name in [('_denoised.nii.gz', 'dwi.nii.gz'), ('_bval.bval', 'gradients.bval'),
                                 ('_bvecs.bvec', 'gradients.bvec'), ('_mask.nii.gz', 'brain.nii.gz')]:
                (folder/(subject+suffix)).rename(folder/name)
        config = self.dataset_config()
        config.update(dwi_pattern='{subject}/dwi.nii.gz', bval_pattern='gradients.bval',
                      bvec_pattern='gradients.bvec', mask_pattern='brain.nii.gz', workers=2)
        self.assertEqual(run_nii2npy(config), [4, 4])

    def test_stale_output_is_reported_and_not_deleted(self):
        self.raw_subject()
        config = self.dataset_config()
        run_nii2npy(config)
        config['bval_min'] = 1500
        with self.assertRaisesRegex(ValueError, 'stale or legacy'):
            run_nii2npy(config)
        self.assertEqual(len(list((self.root/'npy').glob('*.npy'))), 4)

    def test_source_count_and_affine_checks(self):
        with self.assertRaisesRegex(ValueError, 'requires 6 source DWI'):
            check_source_count({'ImageEncoder.ini_conv.weight': torch.empty(64, 6, 3, 3, 3)}, 2)
        config = self.inference_config()
        affine = np.eye(4)
        affine[0, 3] = 5
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), affine), self.root/'b0.nii.gz')
        with self.assertRaisesRegex(ValueError, 'shape/affine'):
            load_input(config)

    def test_padding_mask_empty_patches_and_requested_b0(self):
        config = self.inference_config()
        config['geometry'].update(pad_before=[3, 1, 3], pad_after=[4, 1, 4])
        shape = (9, 10, 11)
        b0 = np.full(shape, 100, np.float32)
        source = np.full((*shape, 2), 20, np.float32)
        mask = np.zeros(shape, np.uint8)
        mask[1:3, 1:3, 1:3] = 1
        for name, values in [('b0', b0), ('source', source), ('mask', mask)]:
            nib.save(nib.Nifti1Image(values, np.eye(4)), self.root/f'{name}.nii.gz')
        config['inference']['mask'] = str(self.root/'mask.nii.gz')
        np.savetxt(self.root/'target.bval', [2000, 0])
        np.savetxt(self.root/'target.bvec', [[1, 0], [0, 0], [0, 0]])
        inputs = load_input(config)
        class ConstantModel(torch.nn.Module):
            def forward(self, target, source, source_sh, size, coords):
                return source.new_full((1, 1, *source.shape[-3:]), 2), source
        result = predict_targets(inputs, ConstantModel(), torch.device('cpu'), config)
        self.assertEqual(result.shape, (*shape, 3))
        np.testing.assert_array_equal(result[..., 0], b0)
        np.testing.assert_array_equal(result[..., 1], mask * 40)
        np.testing.assert_array_equal(result[..., 2], b0)

    def test_actual_model_inference_and_backward(self):
        config = self.inference_config()
        config['model'].update(depth=3, width=16, skips=[1], latent_channels=8, feature_channels=8, decoder_channels=2)
        config['model']['unet']['num_channels'] = [4, 8, 16]
        model = build_model(config, 2)
        torch.save(model.state_dict(), config['inference']['checkpoint'])
        run_inference(config)
        self.assertTrue(np.isfinite(nib.load(self.root/'prediction.nii.gz').get_fdata()).all())
        target = torch.randn(3, 1, 79)
        output, _ = model(target, torch.randn(1, 2, 8, 8, 8), torch.randn(3, 2, 16),
                          torch.tensor([[8]*3]), torch.tensor([[0, 0, 0], [2, 3, 4], [7, 7, 7]]))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.ImageEncoder.ini_conv.weight.grad).all())

    def test_required_paths_and_english_config_comments(self):
        with self.assertRaisesRegex(ValueError, 'set a path for b0'):
            load_config('inference')
        for mode in ('pretraining', 'dataset', 'inference'):
            lines = (ROOT/f'{mode}.ymal').read_text(encoding='utf-8').splitlines()
            for line in lines:
                if line.startswith('#'):
                    self.assertTrue(line.isascii())


if __name__ == '__main__':
    unittest.main()
