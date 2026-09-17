"""Core downstream model prediction from prepared NIfTI inputs."""
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from scipy.ndimage import binary_erosion
from tqdm import tqdm
from ..data.common import load_volume, padding_for_shape
from ..defaults import GEOMETRY
from ..utils.runtime import resolve_device
from .models import DownstreamModel
from .registration import U_Network_feat_all, SpatialTransformer, IntegrateVelocityField


def save_nifti(data, reference, path, dtype=np.float32):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header), str(path))
    print(f'Saved {path}')


def load_patch_input(path, directions, patch_size, source_scale=None):
    image = nib.load(path)
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 4 or data.shape[-1] != directions + 1:
        raise ValueError(f'Input NIfTI must have {directions + 1} volumes: b0 followed by {directions} DWI')
    if not np.isfinite(data).all():
        raise ValueError('Input NIfTI must contain only finite intensities')
    b0, source = data[..., 0], data[..., 1:]
    mask = b0 > 0
    if not mask.any():
        raise ValueError('Input b0 has no positive voxels for prediction')
    b0_scale = float(np.percentile(b0[mask], 99))
    source_scale = float(np.percentile(source[mask], 99)) if source_scale is None else float(source_scale)
    if b0_scale <= 0 or not np.isfinite(source_scale) or source_scale <= 0:
        raise ValueError('Positive b0 and DWI intensities are required for normalization')
    b0 = np.clip(b0, 0, b0_scale) / b0_scale * mask
    source = np.clip(source, 0, source_scale) / source_scale * mask[..., None]
    geometry = {**GEOMETRY, 'patch_size': patch_size}
    padding = padding_for_shape(mask.shape, geometry)
    return image, np.pad(b0, padding), np.pad(source, (*padding, (0, 0))), np.pad(mask, padding), padding


def predict_patch_task(config, model, device):
    image, b0, source, mask, padding = load_patch_input(
        config['input'], config['source_directions'], config['patch_size'])
    subject = tio.Subject(
        source=tio.ScalarImage(tensor=torch.from_numpy(np.moveaxis(source, -1, 0).copy())),
        b0=tio.ScalarImage(tensor=torch.from_numpy(b0[None].astype(np.float32))),
    )
    sampler = tio.inference.GridSampler(subject, tuple(config['patch_size']), tuple(config['patch_overlap']))
    aggregator = tio.inference.GridAggregator(sampler)
    model.eval()
    with torch.inference_mode():
        for patch in tqdm(sampler, desc=f"Predicting {config['task']}"):
            source_patch = patch['source'][tio.DATA].unsqueeze(0).to(device)
            b0_patch = patch['b0'][tio.DATA].to(device)
            prediction = model(source_patch, b0_patch)
            aggregator.add_batch(prediction.cpu(), patch[tio.LOCATION].unsqueeze(0))
    logits = aggregator.get_output_tensor()
    crop = tuple(slice(before, before + size) for (before, _), size in zip(padding, image.shape[:3]))
    if config['task'] == 'parc':
        logits = logits[(slice(None), *crop)]
        maximum, labels = logits.max(dim=0)
        confidence = torch.exp(maximum - torch.logsumexp(logits, dim=0))
        result = labels.numpy().astype(np.int16)
        result[(confidence.numpy() <= 0.1) | ~mask[crop]] = 0
        save_nifti(result, image, config['output'], np.int16)
    else:
        result = logits[0].numpy()[crop] * mask[crop]
        save_nifti(result, image, config['output'])
    return result


def load_feature(path):
    image = nib.load(path)
    features = image.get_fdata(dtype=np.float32)
    if features.ndim != 4 or features.shape[-1] != 64 or not np.isfinite(features).all():
        raise ValueError(f'{path}: expected a finite 4D NIfTI with 64 feature channels')
    return image, features


def center_crop_slices(shape, size):
    if any(length < width for length, width in zip(shape, size)):
        raise ValueError(f'Feature dimensions {shape} are smaller than volume_size {size}')
    # Same asymmetric center crop used by RegistrationDataset.
    return tuple(slice((length - width + 1) // 2, (length - width + 1) // 2 + width)
                 for length, width in zip(shape, size))


def load_registration_input(config):
    moving_image, moving = load_feature(config['moving_feature'])
    fixed_image, fixed = load_feature(config['fixed_feature'])
    if moving.shape != fixed.shape or not np.allclose(moving_image.affine, fixed_image.affine, atol=1e-4):
        raise ValueError('Moving and fixed features must have matching shapes and affines')
    _, b0 = load_volume(config['moving_b0'], moving_image)
    moving_mask = np.any(moving != 0, axis=-1)
    fixed_mask = np.any(fixed != 0, axis=-1)
    if config['mask_erosion']:
        moving_mask = binary_erosion(moving_mask, iterations=config['mask_erosion'])
        fixed_mask = binary_erosion(fixed_mask, iterations=config['mask_erosion'])
    if not moving_mask.any() or not fixed_mask.any():
        raise ValueError('Registration feature support is empty after mask erosion')
    moving *= moving_mask[..., None]
    fixed *= fixed_mask[..., None]
    b0 *= moving_mask
    scale = float(np.percentile(b0, 99))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Moving b0 99th percentile must be positive')
    b0 = np.clip(b0, 0, scale) / scale
    crop = center_crop_slices(b0.shape, config['volume_size'])
    return fixed_image, moving[crop].copy(), fixed[crop].copy(), b0[crop].copy(), crop, scale


def predict_registration(config, model, device):
    reference, moving, fixed, b0, crop, scale = load_registration_input(config)
    model.eval()
    with torch.inference_mode():
        moving = torch.from_numpy(np.moveaxis(moving, -1, 0).copy()).unsqueeze(0).to(device)
        fixed = torch.from_numpy(np.moveaxis(fixed, -1, 0).copy()).unsqueeze(0).to(device)
        b0 = torch.from_numpy(b0).unsqueeze(0).unsqueeze(0).to(device)
        velocity = model(moving, fixed)
        field = IntegrateVelocityField(tuple(config['volume_size']),
            steps=config['integration_steps'], device=device)(velocity)
        warped = SpatialTransformer(config['volume_size']).to(device)(b0, field)
    full_shape = reference.shape[:3]
    flow = np.zeros((*full_shape, 3), dtype=np.float32)
    flow[crop] = np.moveaxis(field[0].cpu().numpy(), 0, -1)
    warped_b0 = np.zeros(full_shape, dtype=np.float32)
    warped_b0[crop] = np.clip(warped[0, 0].cpu().numpy(), 0, 1) * scale
    save_nifti(flow, reference, config['flow_output'])
    save_nifti(warped_b0, reference, config['warped_b0_output'])
    return flow, warped_b0


def run_inference(config):
    device = resolve_device(config['device'], cpu_fallback=True)
    state = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
    if config['task'] == 'reg':
        model = U_Network_feat_all(3, [64, 128, 128, 128], [128, 128, 128, 128, 128, 64, 64])
    else:
        key = 'encoder.ini_conv.weight'
        if key not in state or state[key].shape[1] != config['source_directions']:
            raise ValueError(f"Downstream checkpoint must accept {config['source_directions']} source DWI volumes")
        model = DownstreamModel(config, load_pretrained=False)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    return predict_registration(config, model, device) if config['task'] == 'reg' else predict_patch_task(config, model, device)
