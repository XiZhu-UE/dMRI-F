"""Load one small user configuration and expand stable implementation defaults."""
from copy import deepcopy
from pathlib import Path
import math
from .defaults import MODEL, ENCODING, GEOMETRY, TRAINING

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    'dataset': {'input_dir', 'dwi_pattern', 'bval_pattern', 'bvec_pattern', 'mask_pattern',
                'output_dir', 'metadata_dir', 'bval_min', 'shells', 'shell_tolerance', 'percentile', 'workers'},
    'pretraining': {'data_root', 'metadata_root', 'checkpoint', 'visualization_dir', 'loss_curve',
                    'device', 'source_directions', 'epochs', 'learning_rate', 'train_subjects',
                    'validation_subjects', 'patch_size', 'num_workers', 'normalization_shell',
                    'source_bval_upper', 'perceptual_weight', 'seed'},
    'inference': {'b0', 'source_dwi', 'source_bvals', 'source_bvecs', 'target_bvals', 'target_bvecs',
                  'mask', 'checkpoint', 'output', 'device', 'patch_size', 'patch_overlap',
                  'normalization_scale', 'bval_scale', 'num_workers', 'source_directions'},
}
PATHS = {
    'dataset': ('input_dir', 'output_dir', 'metadata_dir'),
    'pretraining': ('data_root', 'metadata_root', 'checkpoint', 'visualization_dir', 'loss_curve'),
    'inference': ('b0', 'source_dwi', 'source_bvals', 'source_bvecs', 'target_bvals', 'target_bvecs',
                  'mask', 'checkpoint', 'output'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_options(options, mode):
    require(set(options) == FIELDS[mode],
            f'{mode}.ymal: missing keys {sorted(FIELDS[mode] - set(options))}; '
            f'unknown keys {sorted(set(options) - FIELDS[mode])}')
    for key in PATHS[mode]:
        if mode == 'inference' and key == 'mask' and options[key] is None:
            continue
        require(isinstance(options[key], str) and bool(options[key].strip()), f'{mode}.ymal: set a path for {key}')
    key = 'workers' if mode == 'dataset' else 'num_workers'
    require(type(options[key]) is int and options[key] >= (1 if key == 'workers' else 0), f'Invalid {key}')
    if mode == 'dataset':
        require(isinstance(options['dwi_pattern'], str) and options['dwi_pattern'].count('{subject}') == 1,
                'dwi_pattern must contain exactly one {subject} placeholder')
        for key in ('bval_pattern', 'bvec_pattern', 'mask_pattern'):
            require((key == 'mask_pattern' and options[key] is None) or
                    (isinstance(options[key], str) and bool(options[key])), f'Invalid {key}')
        require(options['bval_min'] >= 0 and math.isfinite(options['bval_min']), 'bval_min must be finite and nonnegative')
        require(bool(options['shells']) and all(type(v) is int and v > 0 for v in options['shells']),
                'shells must contain positive integer b-values')
        require(len(set(options['shells'])) == len(options['shells']), 'shell centers must be unique')
        require(options['shell_tolerance'] >= 0 and math.isfinite(options['shell_tolerance']), 'Invalid shell_tolerance')
        require(0 < options['percentile'] <= 100, 'percentile must be in (0, 100]')
    else:
        patch = options['patch_size']
        require(len(patch) == 3 and all(type(v) is int and v >= 8 and v % 4 == 0 for v in patch),
                'patch_size must contain three multiples of 4, each at least 8')
        if mode == 'pretraining':
            for key in ('source_directions', 'epochs', 'train_subjects', 'validation_subjects'):
                require(type(options[key]) is int and options[key] > 0, f'{key} must be a positive integer')
            for key in ('learning_rate', 'normalization_shell', 'source_bval_upper'):
                require(math.isfinite(options[key]) and options[key] > 0, f'{key} must be positive and finite')
            require(math.isfinite(options['perceptual_weight']) and options['perceptual_weight'] >= 0,
                    'perceptual_weight must be finite and nonnegative')
            require(options['seed'] is None or type(options['seed']) is int, 'seed must be an integer or null')
        else:
            require(type(options['source_directions']) is int and options['source_directions'] > 0,
                    'source_directions must be a positive integer')
            overlap = options['patch_overlap']
            require(len(overlap) == 3 and all(type(v) is int and 0 <= v < p and v % 2 == 0
                                            for v, p in zip(overlap, patch)),
                    'patch_overlap must contain even values smaller than patch_size')
            require(math.isfinite(options['bval_scale']) and options['bval_scale'] > 0, 'bval_scale must be positive')
            scale = options['normalization_scale']
            require(scale is None or (math.isfinite(scale) and scale > 0), 'normalization_scale must be positive or null')


def load_config(mode='pretraining', path=None):
    import yaml
    if mode not in FIELDS:
        raise ValueError(f'Unknown configuration: {mode}')
    path = Path(path) if path is not None else ROOT / f'{mode}.ymal'
    with path.open(encoding='utf-8') as stream:
        options = yaml.safe_load(stream)
    require(isinstance(options, dict), 'Configuration must be a YAML mapping')
    try:
        validate_options(options, mode)
    except (TypeError, KeyError) as error:
        raise ValueError(f'Malformed {mode} configuration: {error}') from error
    for key in PATHS[mode]:
        value = options[key]
        if value is not None and not value.startswith('/') and not Path(value).is_absolute():
            options[key] = str((path.resolve().parent / value).resolve())
    if mode == 'dataset':
        return options
    config = {'model': deepcopy(MODEL), 'encoding': deepcopy(ENCODING), 'geometry': deepcopy(GEOMETRY)}
    config['geometry']['patch_size'] = options.pop('patch_size')
    if mode == 'pretraining':
        settings = deepcopy(TRAINING)
        settings['loader']['num_workers'] = options.pop('num_workers')
        settings.update(options)
        settings['visualization_slice'] = config['geometry']['patch_size'][0] // 2
        config['training'] = settings
        config['files'] = {'mask_suffix': '_mask.nii.gz', 'xmax_suffix': '_xmax.json'}
    else:
        config['inference'] = options
    return config
