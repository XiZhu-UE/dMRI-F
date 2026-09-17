"""Training and validation with the original L1 + perceptual objective."""
import gc
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from generative.losses import PerceptualLoss
from tqdm import tqdm
from .data.training import Generalized_NeRF_Dataset
from .models.encoding import get_embedder
from .models.nerf import build_model
from .utils.runtime import loader_kwargs, resolve_device, seed_everything
from .utils.plotting import save_preview, save_loss_curve


def run_epoch(model, loader, embed_pos, device, perceptual_loss, weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {'loss': 0.0, 'recon_loss': 0.0, 'perceptual_loss': 0.0}
    count = 0
    last_preview = None
    reconstruction_loss = nn.L1Loss()
    progress = tqdm(loader, ncols=220, desc='Train' if training else 'Validation')
    with torch.set_grad_enabled(training):
        for batch in progress:
            valid_coords = batch['coord_local'].to(device).squeeze(0)
            # Empty background patches carry no supervised voxels.
            if valid_coords.numel() == 0:
                continue
            source = batch['source_patch'].to(device)
            target = batch['target_patch'].to(device)
            source_sh = batch['source_bvec_sh_encoding'].to(device).squeeze(0)
            target_sh = batch['target_bvec_sh_encoding'].to(device).squeeze(0)
            coord_norm = batch['coord_norm'].to(device).squeeze(0).unsqueeze(1)
            target_feature = torch.cat([embed_pos(coord_norm), target_sh], -1)
            prediction, features = model(target_feature, source, source_sh, batch['image_size'].to(device), valid_coords)
            recon = reconstruction_loss(prediction, target)
            perceptual = perceptual_loss(prediction, target)
            loss = recon + weight * perceptual
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            for key, value in zip(totals, (loss, recon, perceptual)):
                totals[key] += value.item()
            count += 1
            progress.set_postfix({key: value / count for key, value in totals.items()})
            if not training:
                last_preview = tuple(t.detach() for t in (source, target, prediction, features))
    if count == 0:
        raise RuntimeError('No nonempty patches were processed; check masks and data paths')
    return {key: value / count for key, value in totals.items()}, last_preview


def run_training(config):
    settings = config['training']
    seed_everything(settings['seed'])
    device = resolve_device(settings['device'])
    datasets = [Generalized_NeRF_Dataset(
        settings['source_directions'], settings['data_root'], is_Train=is_train,
        bvec_encoding_level=config['encoding']['spherical_harmonic_levels'], config=config,
    ) for is_train in (True, False)]
    train_loader = DataLoader(datasets[0], shuffle=settings['train_shuffle'], **loader_kwargs(settings['loader']))
    val_loader = DataLoader(datasets[1], shuffle=settings['validation_shuffle'], **loader_kwargs(settings['loader']))
    model = build_model(config, settings['source_directions']).to(device)
    checkpoint = Path(settings['checkpoint'])
    if settings['resume_if_exists'] and checkpoint.exists():
        # Original checkpoints contain weights only, not optimizer/epoch state.
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    Path(settings['visualization_dir']).mkdir(parents=True, exist_ok=True)
    Path(settings['loss_curve']).parent.mkdir(parents=True, exist_ok=True)
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type=settings['perceptual_network']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    embed_pos, _ = get_embedder(config['encoding']['position_frequencies'])
    train_history, val_history = [], []
    for epoch in range(settings['epochs']):
        train_metrics, _ = run_epoch(model, train_loader, embed_pos, device, perceptual_loss, settings['perceptual_weight'], optimizer)
        val_metrics, preview = run_epoch(model, val_loader, embed_pos, device, perceptual_loss, settings['perceptual_weight'])
        train_history.append(train_metrics['loss'])
        val_history.append(val_metrics['loss'])
        save_preview(*preview, epoch, settings)
        torch.save(model.state_dict(), checkpoint)
        del preview
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    save_loss_curve(train_history, val_history, settings)
