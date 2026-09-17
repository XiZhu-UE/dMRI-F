"""Load task-specific YAML settings without command-line arguments."""
from pathlib import Path
import yaml


def load_config(task, path=None):
    if task not in ('parc', 'syn', 'reg'):
        raise ValueError(f'Unknown downstream task: {task}')
    default = Path(__file__).resolve().parents[2] / f'downstream_{task}_train.ymal'
    path = Path(path or default).resolve()
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f'{path}: expected a configuration mapping')
    config['task'] = task
    for key, value in list(config.items()):
        if key.endswith('_dir') or key.endswith('_checkpoint') or key == 'checkpoint':
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{key} must be a nonempty path')
            candidate = Path(value).expanduser()
            config[key] = str((path.parent / candidate).resolve())
    for key in ('epochs', 'batch_size', 'samples_per_subject'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['num_workers'] < 0 or config['learning_rate'] <= 0:
        raise ValueError('num_workers must be nonnegative and learning_rate positive')
    for key in ('train_slice', 'validation_slice'):
        value = config[key]
        if not isinstance(value, list) or len(value) != 2 or any(
                item is not None and type(item) is not int for item in value):
            raise ValueError(f'{key} must be [start, stop], using null for an open endpoint')
    shape = config['volume_size'] if task == 'reg' else config['patch_size']
    divisor = 16 if task == 'reg' else (8 if task == 'syn' else 4)
    if len(shape) != 3 or any(type(n) is not int or n < divisor or n % divisor for n in shape):
        raise ValueError(f'Spatial sizes must contain three positive multiples of {divisor}')
    if task == 'reg':
        if config['mask_erosion'] < 0 or config['integration_steps'] < 0:
            raise ValueError('mask_erosion and integration_steps must be nonnegative')
    elif config['source_directions'] < 1:
        raise ValueError('source_directions must be positive')
    return config
