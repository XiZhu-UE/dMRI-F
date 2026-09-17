"""Training inputs shared by the downstream tasks."""
from collections import defaultdict
from pathlib import Path
import re
import warnings
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from torch.utils.data import Dataset
from ..data.common import load_volume, padding_for_shape
from ..data.shells import load_xmax
from ..defaults import GEOMETRY


NPY_PATTERN = re.compile(
    r'(?P<subject>[A-Za-z0-9_.-]+)_bval_(?P<bval>[\d.]+)_bvec_'
    r'[-\d.]+_[-\d.]+_[-\d.]+(?:_vol_\d+)?\.npy')


def subject_path(config, kind, subject):
    return Path(config[f'{kind}_dir']) / config[f'{kind}_pattern'].format(subject=subject)


def select_subjects(subjects, config, training):
    subjects = sorted(subjects)
    train = subjects[slice(*config['train_slice'])]
    validation = subjects[slice(*config['validation_slice'])]
    if training and set(train) & set(validation):
        warnings.warn('Training and validation subjects overlap; adjust the configured slices.', stacklevel=2)
    selected = train if training else validation
    if not selected:
        raise ValueError('Empty subject split; check input paths and train_slice/validation_slice')
    return selected


def normalize(data, values, context):
    if not values.size:
        raise ValueError(f'{context}: empty normalization region')
    maximum = np.percentile(values, 99)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError(f'{context}: the 99th percentile must be positive and finite')
    return np.clip(data, 0, maximum) / maximum


class PatchDataset(Dataset):
    def __init__(self, config, training=True):
        self.config = config
        paths = defaultdict(list)
        for path in sorted(Path(config['npy_dir']).glob('*.npy')):
            match = NPY_PATTERN.fullmatch(path.name)
            if match:
                paths[match['subject']].append((float(match['bval']), path))
        subjects = paths.keys()
        if config['task'] == 'parc':
            subjects = [p.name[:-len('_xmax.json')] for p in
                        Path(config['metadata_dir']).glob('*_xmax.json')]
        self.subjects = select_subjects(subjects, config, training)
        self.records = {}
        for subject in self.subjects:
            source = [p for b, p in sorted(paths[subject]) if b < config['source_bval_upper']]
            if len(source) < config['source_directions']:
                raise ValueError(f'{subject}: too few source DWI volumes below source_bval_upper')
            reference, mask = load_volume(subject_path(config, 'mask', subject))
            _, b0 = load_volume(subject_path(config, 'b0', subject), reference)
            _, target = load_volume(subject_path(config, 'target', subject), reference)
            b0 = b0 * mask
            b0 = normalize(b0, b0[mask > 0], subject)
            padding = padding_for_shape(mask.shape, {**GEOMETRY, 'patch_size': config['patch_size']})
            if config['task'] == 'parc':
                if (target < 0).any() or not np.equal(target, np.floor(target)).all():
                    raise ValueError(f'{subject}: parcellation labels must be nonnegative integers')
                target[target == config['background_label']] = 0
                # Preserve the original per-subject contiguous label mapping.
                labels, target = np.unique(target, return_inverse=True)
                target = target.reshape(mask.shape).astype(np.int64)
                if len(labels) > config['num_classes']:
                    raise ValueError(f'{subject}: more labels than num_classes')
                target = np.pad(target, padding)
            else:
                target = np.pad(target * mask, padding)
                target = normalize(target, target, subject)
            scales = load_xmax(Path(config['metadata_dir']) / f'{subject}_xmax.json')
            scale = scales[config['normalization_shell']]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'{subject}: invalid shell normalization scale')
            self.records[subject] = dict(paths=source, mask=np.pad(mask, padding),
                b0=np.pad(b0, padding), target=target, padding=padding, scale=scale,
                original_shape=mask.shape)

    def __len__(self):
        return self.config['samples_per_subject'] * len(self.subjects)

    def __getitem__(self, index):
        record = self.records[np.random.choice(self.subjects)]
        chosen = np.random.choice(len(record['paths']), self.config['source_directions'], replace=False)
        volumes = [np.load(record['paths'][i], mmap_mode='r') for i in chosen]
        if any(v.shape != record['original_shape'] or not np.isfinite(v).all() for v in volumes):
            raise ValueError('Source NPY volumes must be finite and match the mask shape')
        source = np.pad(np.stack(volumes, axis=-1), (*record['padding'], (0, 0)))
        source = np.clip(source * record['mask'][..., None], 0, record['scale']) / record['scale']
        patch = self.config['patch_size']
        starts = [np.random.randint(n - p + 1) for n, p in zip(record['mask'].shape, patch)]
        crop = tuple(slice(s, s + p) for s, p in zip(starts, patch))
        return dict(source=np.moveaxis(source[crop], -1, 0).astype(np.float32),
                    b0=record['b0'][crop].astype(np.float32),
                    mask=record['mask'][crop].astype(np.float32),
                    target=record['target'][crop].copy())


class RegistrationDataset(Dataset):
    def __init__(self, config, training=True):
        self.config = config
        pattern = config['feature_pattern']
        if pattern.count('{subject}') != 1:
            raise ValueError('feature_pattern must contain one {subject} placeholder')
        matcher = re.compile(re.escape(pattern.replace('\\', '/')).replace(
            re.escape('{subject}'), '(?P<subject>[^/]+)'))
        self.features = {}
        root = Path(config['feature_dir'])
        for path in sorted(root.glob(pattern.replace('{subject}', '*'))):
            match = matcher.fullmatch(path.relative_to(root).as_posix())
            if path.is_file() and match:
                self.features[match['subject']] = path
        self.subjects = select_subjects(self.features, config, training)

    def __len__(self):
        return self.config['samples_per_subject'] * len(self.subjects)

    def _load(self, subject):
        image = nib.load(str(self.features[subject]))
        features = image.get_fdata(dtype=np.float32)
        if features.ndim != 4 or features.shape[-1] != 64 or not np.isfinite(features).all():
            raise ValueError(f'{subject}: expected finite features with shape (X, Y, Z, 64)')
        _, b0 = load_volume(subject_path(self.config, 'b0', subject), image)
        _, mask = load_volume(subject_path(self.config, 'mask', subject), image)
        if self.config['mask_erosion']:
            mask = binary_erosion(mask, iterations=self.config['mask_erosion'])
        features *= mask[..., None]
        b0 *= mask
        b0 = normalize(b0, b0, subject)
        size = self.config['volume_size']
        if any(n < s for n, s in zip(b0.shape, size)):
            raise ValueError(f'{subject}: input is smaller than volume_size')
        starts = [(n - s + 1) // 2 for n, s in zip(b0.shape, size)]
        crop = tuple(slice(start, start + s) for start, s in zip(starts, size))
        return np.moveaxis(features[crop], -1, 0).copy(), b0[crop].copy()

    def __getitem__(self, index):
        moving, fixed = [self._load(np.random.choice(self.subjects)) for _ in range(2)]
        return dict(moving_features=moving[0], fixed_features=fixed[0],
                    moving_b0=moving[1], fixed_b0=fixed[1])
