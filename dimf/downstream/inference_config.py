"""Load the three compact downstream inference configurations."""
from pathlib import Path
import yaml


COMMON = {'checkpoint', 'device'}
PATCH = COMMON | {'input', 'output', 'source_directions', 'patch_size', 'patch_overlap'}
REG = COMMON | {'moving_feature', 'fixed_feature', 'moving_b0', 'flow_output',
                'warped_b0_output', 'volume_size', 'integration_steps', 'mask_erosion'}
PATHS = {'input', 'output', 'checkpoint', 'moving_feature', 'fixed_feature',
         'moving_b0', 'flow_output', 'warped_b0_output'}


def load_inference_config(task, path=None):
    if task not in ('parc', 'syn', 'reg'):
        raise ValueError(f'Unknown downstream task: {task}')
    default = Path(__file__).resolve().parents[2] / f'downstream_{task}_inference.ymal'
    path = Path(path or default).resolve()
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    expected = REG if task == 'reg' else PATCH
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f'{path}: expected keys {sorted(expected)}')
    config['task'] = task
    for key in expected & PATHS:
        value = config[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{key} must be a nonempty path')
        candidate = Path(value).expanduser()
        config[key] = str((path.parent / candidate).resolve())
        if key in ('output', 'flow_output', 'warped_b0_output') and not (
                value.endswith('.nii') or value.endswith('.nii.gz')):
            raise ValueError(f'{key} must end in .nii or .nii.gz')
    if not isinstance(config['device'], str) or not config['device'].strip():
        raise ValueError('device must be a nonempty PyTorch device')
    if task == 'reg':
        shape = config['volume_size']
        for name in ('integration_steps', 'mask_erosion'):
            if type(config[name]) is not int or config[name] < 0:
                raise ValueError(f'{name} must be nonnegative')
        if config['flow_output'] == config['warped_b0_output']:
            raise ValueError('flow_output and warped_b0_output must differ')
        divisor = 16
    else:
        shape = config['patch_size']
        if type(config['source_directions']) is not int or config['source_directions'] < 1:
            raise ValueError('source_directions must be positive')
        divisor = 4 if task == 'parc' else 8
    if not isinstance(shape, list) or len(shape) != 3 or any(
            type(n) is not int or n < divisor or n % divisor for n in shape):
        raise ValueError(f'spatial size must contain three multiples of {divisor}')
    if task != 'reg':
        overlap = config['patch_overlap']
        if not isinstance(overlap, list) or len(overlap) != 3 or any(
                type(v) is not int or v < 0 or v >= size or v % 2
                for v, size in zip(overlap, shape)):
            raise ValueError('patch_overlap must contain three even values below patch_size')
    return config
