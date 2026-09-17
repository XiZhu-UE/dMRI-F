"""Save training diagnostics without opening interactive windows."""
import numpy as np


def save_figures(history, preview, output, task):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots()
    for split in ('train', 'validation'):
        axis.plot([row['epoch'] for row in history],
                  [row[f'{split}_loss'] for row in history], label=split)
    axis.set(xlabel='Epoch', ylabel='Loss')
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / 'loss_curve.png')
    plt.close(fig)
    if preview is None:
        return
    moving, target, prediction, field = preview
    arrays = [item[0, 0].numpy() for item in (moving, target, prediction)]
    axis_index = 2 if task == 'reg' else 0
    location = arrays[0].shape[axis_index] // 2
    images = [np.take(item, location, axis=axis_index) for item in arrays]
    images.append(np.abs(images[2] - images[1]))
    titles = ['Input', 'Target', 'Prediction', 'Absolute error']
    if field is not None:
        vectors = field[0, :, :, :, location].numpy()
        rgb = np.moveaxis(vectors, 0, -1)
        rgb = rgb - rgb.min(axis=(0, 1), keepdims=True)
        rgb /= np.maximum(rgb.max(axis=(0, 1), keepdims=True), 1e-8)
        images.append(rgb)
        titles.append('Displacement (RGB)')
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for axis, item, title in zip(axes, images, titles):
        axis.imshow(item, cmap='viridis' if title == 'Absolute error' else 'gray')
        axis.set_title(title)
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(output / f"epoch_{history[-1]['epoch']:04d}.png")
    plt.close(fig)
