"""Shared training loop with task-specific models and objectives."""
import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import PatchDataset, RegistrationDataset
from .models import DownstreamModel
from .parcellation_loss import CombinedLoss3D_PresentDice
from .registration import (U_Network_feat_all, SpatialTransformer,
                           IntegrateVelocityField, ncc_loss, gradient_loss)
from .plotting import save_figures
from ..utils.runtime import loader_kwargs, resolve_device, seed_everything


class Objective:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        task = config['task']
        if task == 'parc':
            self.loss = CombinedLoss3D_PresentDice(
                weight_dice=config['dice_weight'], weight_ce=config['ce_weight'],
                ignore_index=-1).to(device)
        elif task == 'syn':
            self.perceptual = None
            if config['perceptual_weight']:
                from generative.losses import PerceptualLoss
                self.perceptual = PerceptualLoss(spatial_dims=3, network_type='alex').to(device)
                self.perceptual.requires_grad_(False)
        else:
            self.transformer = SpatialTransformer(config['volume_size']).to(device)
            self.integrator = IntegrateVelocityField(tuple(config['volume_size']),
                steps=config['integration_steps'], device=device)

    def __call__(self, model, batch):
        task = self.config['task']
        batch = {key: value.to(self.device) for key, value in batch.items()}
        if task == 'reg':
            velocity = model(batch['moving_features'].float(), batch['fixed_features'].float())
            field = self.integrator(velocity)
            moving = batch['moving_b0'].float().unsqueeze(1)
            target = batch['fixed_b0'].float().unsqueeze(1)
            prediction = self.transformer(moving, field)
            similarity = ncc_loss(prediction, target, 1, self.device)
            regularity = gradient_loss(velocity)
            loss = similarity + self.config['gradient_weight'] * regularity
            metrics = {'ncc': similarity.item(), 'gradient': regularity.item()}
            preview = (moving, target, prediction, field)
        else:
            source, b0 = batch['source'].float(), batch['b0'].float()
            mask = batch['mask'].float().unsqueeze(1)
            prediction = model(source, b0) * mask
            if task == 'parc':
                loss, dice, ce = self.loss(prediction, batch['target'].long())
                metrics = {'dice': dice.item(), 'ce': ce.item()}
                preview = None
            else:
                target = batch['target'].float().unsqueeze(1)
                valid = mask > 0
                if not valid.any():
                    raise ValueError('Sampled synthesis patch has an empty mask; reduce patch sampling background')
                reconstruction = F.l1_loss(prediction[valid], target[valid])
                perceptual = self.perceptual(prediction, target) if self.perceptual else prediction.new_zeros(())
                loss = reconstruction + self.config['perceptual_weight'] * perceptual
                metrics = {'l1': reconstruction.item(), 'perceptual': perceptual.item()}
                preview = (source[:, :1], target, prediction, None)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite {task} loss')
        if preview is not None:
            preview = tuple(item.detach().cpu() if item is not None else None for item in preview)
        return loss, {'loss': loss.item(), **metrics}, preview


def run_epoch(model, loader, objective, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals, preview = {}, None
    progress = tqdm(loader, desc='Train' if training else 'Validation')
    with torch.set_grad_enabled(training):
        for batch in progress:
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics, preview = objective(model, batch)
            if training:
                loss.backward()
                optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            progress.set_postfix(loss=metrics['loss'])
    return {key: value / len(loader) for key, value in totals.items()}, preview


def run_training(config):
    seed_everything(config['seed'])
    device = resolve_device(config['device'])
    dataset_class = RegistrationDataset if config['task'] == 'reg' else PatchDataset
    datasets = [dataset_class(config, training=training) for training in (True, False)]
    options = loader_kwargs(dict(batch_size=config['batch_size'], num_workers=config['num_workers'],
        prefetch_factor=2, persistent_workers=True, pin_memory=device.type == 'cuda'))
    loaders = [DataLoader(dataset, shuffle=training or config['task'] != 'reg', **options)
               for dataset, training in zip(datasets, (True, False))]
    checkpoint = Path(config['checkpoint'])
    resume = config['resume_if_exists'] and checkpoint.is_file()
    if config['task'] == 'reg':
        model = U_Network_feat_all(3, [64, 128, 128, 128], [128, 128, 128, 128, 128, 64, 64])
    else:
        model = DownstreamModel(config, load_pretrained=not resume)
    if resume:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        print(f'Loaded model weights from {checkpoint}; starting a new optimizer.')
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=config['learning_rate'])
    objective = Objective(config, device)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, config['epochs'] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")
        training, _ = run_epoch(model, loaders[0], objective, optimizer)
        validation, preview = run_epoch(model, loaders[1], objective)
        row = {'epoch': epoch, **{f'train_{k}': v for k, v in training.items()},
               **{f'validation_{k}': v for k, v in validation.items()}}
        history.append(row)
        torch.save(model.state_dict(), checkpoint)
        with (output / 'loss_history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        save_figures(history, preview, output, config['task'])
    return history
