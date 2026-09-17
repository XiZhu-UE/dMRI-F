"""Generate the pretrained encoder's 64-channel feature volume."""
from pathlib import Path
import numpy as np
import torch
import torchio as tio
from tqdm import tqdm
import yaml
from .defaults import MODEL
from .downstream.inference import load_patch_input, save_nifti
from .models.unet import Unet_3D
from .utils.runtime import resolve_device


def load_config(path=None):
    path = Path(path or Path(__file__).resolve().parents[1] / 'feature_generate.ymal').resolve()
    with path.open(encoding='utf-8') as stream:
        settings = yaml.safe_load(stream)
    expected = {'input', 'output', 'checkpoint', 'device', 'source_directions',
                'patch_size', 'patch_overlap', 'normalization_scale'}
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError(f'{path}: expected keys {sorted(expected)}')
    for key in ('input', 'output', 'checkpoint'):
        value = settings[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{key} must be a nonempty path')
        settings[key] = str((path.parent / Path(value).expanduser()).resolve())
    if not (settings['output'].endswith('.nii') or settings['output'].endswith('.nii.gz')):
        raise ValueError('output must be a .nii or .nii.gz path')
    if not isinstance(settings['device'], str) or not settings['device'].strip():
        raise ValueError('device must be a nonempty PyTorch device')
    if type(settings['source_directions']) is not int or settings['source_directions'] != 6:
        raise ValueError('This example expects six DWI volumes after b0')
    size = settings['patch_size']
    if not isinstance(size, list) or len(size) != 3 or any(
            type(n) is not int or n < 8 or n % 4 for n in size):
        raise ValueError('patch_size must contain three multiples of 4, at least 8')
    overlap = settings['patch_overlap']
    if not isinstance(overlap, list) or len(overlap) != 3 or any(
            type(n) is not int or n < 0 or n >= p or n % 2
            for n, p in zip(overlap, size)):
        raise ValueError('patch_overlap must contain three even values below patch_size')
    scale = settings['normalization_scale']
    if scale is not None and (not isinstance(scale, (int, float)) or
                              not np.isfinite(scale) or scale <= 0):
        raise ValueError('normalization_scale must be a positive number or null')
    return settings


def build_encoder(checkpoint, source_directions):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    prefix = 'ImageEncoder.'
    weights = {name[len(prefix):]: value for name, value in state.items()
               if name.startswith(prefix)}
    first = weights.get('ini_conv.weight')
    if first is None or first.shape[1] != source_directions:
        raise ValueError(f'Checkpoint must contain an ImageEncoder for {source_directions} DWI volumes')
    encoder = Unet_3D(in_channel=source_directions,
                      out_channel=MODEL['latent_channels'], **MODEL['unet'])
    encoder.load_state_dict(weights, strict=True)
    return encoder


def generate_features(settings, encoder, device):
    image, _, source, mask, padding = load_patch_input(
        settings['input'], settings['source_directions'], settings['patch_size'],
        settings['normalization_scale'])
    subject = tio.Subject(source=tio.ScalarImage(
        tensor=torch.from_numpy(np.moveaxis(source, -1, 0).copy())))
    sampler = tio.inference.GridSampler(subject, tuple(settings['patch_size']),
                                        tuple(settings['patch_overlap']))
    aggregator = tio.inference.GridAggregator(sampler)
    encoder.eval()
    with torch.inference_mode():
        for patch in tqdm(sampler, desc='Generating features'):
            volume = patch['source'][tio.DATA].unsqueeze(0).to(device)
            encoded = encoder(volume)
            if encoded.shape[1] != 64:
                raise ValueError(f'Encoder output has {encoded.shape[1]} channels; expected 64')
            aggregator.add_batch(encoded.cpu(), patch[tio.LOCATION].unsqueeze(0))
    crop = tuple(slice(before, before + length)
                 for (before, _), length in zip(padding, image.shape[:3]))
    features = np.moveaxis(aggregator.get_output_tensor().numpy()[(slice(None), *crop)], 0, -1)
    features *= mask[crop][..., None]
    save_nifti(features, image, settings['output'])
    return features


def run(settings):
    device = resolve_device(settings['device'], cpu_fallback=True)
    encoder = build_encoder(settings['checkpoint'], settings['source_directions']).to(device)
    return generate_features(settings, encoder, device)
