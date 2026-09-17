"""Generate requested target DWI volumes from a separate source acquisition."""
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
import torchio as tio
from tqdm import tqdm
from .data.inference import load_input
from .models.nerf import build_model
from .models.encoding import get_embedder, components_from_spherical_harmonics
from .utils.runtime import resolve_device


def check_source_count(state, count):
    key = 'ImageEncoder.ini_conv.weight'
    if key not in state:
        raise ValueError(f'Checkpoint is not a compatible model state_dict: missing {key}')
    expected = state[key].shape[1]
    if count != expected:
        raise ValueError(f'Checkpoint requires {expected} source DWI volumes, but {count} were supplied; '
                         'use matching source data or a checkpoint trained for this count')


def encode_directions(bvals, bvecs, config):
    directions = bvecs * np.sqrt(bvals / config['inference']['bval_scale'])[:, None]
    return components_from_spherical_harmonics(
        config['encoding']['spherical_harmonic_levels'], torch.as_tensor(directions, dtype=torch.float32))


def predict_targets(inputs, model, device, config):
    """Return b0 + targets in the target table's exact order and original space."""
    settings = config['inference']
    normalized = inputs.source / inputs.scale * inputs.mask[..., None]
    padded_source = np.pad(normalized, inputs.padding + ((0, 0),))
    padded_mask = np.pad(inputs.mask, inputs.padding)
    full_shape = np.asarray(padded_mask.shape, dtype=np.float32)
    subject = tio.Subject(
        dwi=tio.ScalarImage(tensor=torch.from_numpy(padded_source).permute(3, 0, 1, 2)),
        mask=tio.LabelMap(tensor=torch.from_numpy(padded_mask.astype(np.uint8))[None]),
    )
    sampler = tio.inference.GridSampler(subject, tuple(config['geometry']['patch_size']), tuple(settings['patch_overlap']))
    loader = tio.SubjectsLoader(sampler, batch_size=1, num_workers=settings['num_workers'], pin_memory=device.type == 'cuda')
    # Metadata stays outside TorchIO's collation, which may return arrays or tensors depending on version.
    source_sh = encode_directions(inputs.source_bvals, inputs.source_bvecs, config).to(device)
    target_sh = encode_directions(inputs.target_bvals, inputs.target_bvecs, config).to(device)
    target_indices = np.flatnonzero(inputs.target_bvals > 0)
    aggregators = {int(index): tio.inference.GridAggregator(sampler) for index in target_indices}
    embed, _ = get_embedder(config['encoding']['position_frequencies'])
    patch_size_tensor = torch.tensor([config['geometry']['patch_size']], dtype=torch.float32, device=device)
    with torch.inference_mode():
        if len(target_indices):
            for patch in tqdm(loader, desc='Predicting target DWI'):
                locations = patch[tio.LOCATION]
                mask = patch['mask'][tio.DATA][0, 0].cpu().numpy()
                coords = np.argwhere(mask > 0)
                source = patch['dwi'][tio.DATA].to(device)
                if len(coords):
                    global_coords = coords + locations[0, :3].cpu().numpy()
                    coord_norm = torch.as_tensor(2 * global_coords / (full_shape - 1) - 1,
                                                 dtype=torch.float32, device=device).unsqueeze(1)
                    coord_feature = embed(coord_norm)
                    coords_tensor = torch.as_tensor(coords, device=device)
                    source_feature = source_sh.unsqueeze(0).expand(len(coords), -1, -1)
                for index in target_indices:
                    if len(coords):
                        target_feature = torch.cat([coord_feature, target_sh[index].view(1, 1, -1).expand(len(coords), 1, -1)], -1)
                        logits, _ = model(target_feature, source, source_feature, patch_size_tensor, coords_tensor)
                    else:
                        logits = source.new_zeros((1, 1, *config['geometry']['patch_size']))
                    aggregators[int(index)].add_batch(logits.cpu(), locations)
    result = np.empty((*inputs.b0.shape, len(inputs.target_bvals) + 1), dtype=np.float32)
    result[..., 0] = inputs.b0
    crop = tuple(slice(low, low + size) for (low, _), size in zip(inputs.padding, inputs.b0.shape))
    for index, bval in enumerate(inputs.target_bvals):
        if bval == 0:
            result[..., index + 1] = inputs.b0
        else:
            prediction = aggregators[index].get_output_tensor()[0].numpy()[crop]
            result[..., index + 1] = prediction * inputs.scale * inputs.mask
    return result


def save_prediction(result, inputs, output):
    output = Path(output)
    if not (output.name.endswith('.nii.gz') or output.name.endswith('.nii')):
        raise ValueError('Inference output must end in .nii or .nii.gz')
    output.parent.mkdir(parents=True, exist_ok=True)
    header = inputs.image.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(result, inputs.image.affine, header)
    nib.save(image, str(output))
    stem = output.name[:-7] if output.name.endswith('.nii.gz') else output.stem
    bvals = np.r_[0.0, inputs.target_bvals]
    bvecs = np.vstack([np.zeros(3), inputs.target_bvecs])
    bvecs[bvals == 0] = 0
    np.savetxt(output.with_name(stem + '.bval'), bvals[None], fmt='%.10g')
    np.savetxt(output.with_name(stem + '.bvec'), bvecs.T, fmt='%.10g')
    print(f'Saved {output}: 1 input b0 + {len(inputs.target_bvals)} requested target volumes.')


def run_inference(config):
    settings = config['inference']
    inputs = load_input(config)
    device = resolve_device(settings['device'], cpu_fallback=True)
    state = torch.load(settings['checkpoint'], map_location='cpu', weights_only=True)
    check_source_count(state, inputs.source.shape[-1])
    model = build_model(config, inputs.source.shape[-1]).to(device)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    result = predict_targets(inputs, model, device, config)
    save_prediction(result, inputs, settings['output'])
