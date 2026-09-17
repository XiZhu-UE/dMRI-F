from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_preview(source_patch, target_patch, pred_patch, image_feature, epoch, settings):
    img_input = source_patch[0, 0].cpu().numpy()
    img_gt = target_patch[0, 0].cpu().numpy()
    img_pred = pred_patch[0, 0].cpu().numpy()
    diff = np.abs(img_pred - img_gt)

    fig, ax = plt.subplots(2, 4, figsize=(24, 10))  # 加宽画布防止colorbar重叠
    plt.subplots_adjust(wspace=0.4)  # 调整子图间距

    # input image
    im_input = ax[0,0].imshow(img_input[settings['visualization_slice'], :, :], cmap='gray')
    ax[0,0].set_title('Input Image')
    fig.colorbar(im_input, ax=ax[0,0], shrink=0.8)

    # Ground Truth
    im_gt = ax[0,1].imshow(img_gt[settings['visualization_slice'], :, :], cmap='gray')
    ax[0,1].set_title('Ground Truth')
    fig.colorbar(im_gt, ax=ax[0,1], shrink=0.8)  # 添加colorbar

    # Predicted
    im_pred = ax[0,2].imshow(img_pred[settings['visualization_slice'], :, :], cmap='gray')
    ax[0,2].set_title('Predicted')
    fig.colorbar(im_pred, ax=ax[0,2], shrink=0.8)

    # Difference
    im_diff = ax[0,3].imshow(diff[settings['visualization_slice'], :, :], cmap='viridis')  # 使用彩色colormap更直观
    ax[0,3].set_title('Difference Map')
    cbar = fig.colorbar(im_diff, ax=ax[0,3], shrink=0.8)
    cbar.set_label('Absolute Error')  # 设置colorbar标签

    # Image Feature
    im_feature = ax[1,0].imshow(image_feature[0, settings['visualization_channels'][0], settings['visualization_slice'], :, :].cpu().numpy(), cmap='gray')
    ax[1,0].set_title('Image Feature')
    fig.colorbar(im_feature, ax=ax[1,0], shrink=0.8)

    im_feature = ax[1, 1].imshow(image_feature[0, settings['visualization_channels'][1], settings['visualization_slice'], :, :].cpu().numpy(), cmap='gray')
    ax[1, 1].set_title('Image Feature')
    fig.colorbar(im_feature, ax=ax[1, 1], shrink=0.8)

    im_feature = ax[1, 2].imshow(image_feature[0, settings['visualization_channels'][2], settings['visualization_slice'], :, :].cpu().numpy(), cmap='gray')
    ax[1, 2].set_title('Image Feature')
    fig.colorbar(im_feature, ax=ax[1, 2], shrink=0.8)

    im_feature = ax[1, 3].imshow(image_feature[0, settings['visualization_channels'][3], settings['visualization_slice'], :, :].cpu().numpy(), cmap='gray')
    ax[1, 3].set_title('Image Feature')
    fig.colorbar(im_feature, ax=ax[1, 3], shrink=0.8)

    plt.savefig(
        str(Path(settings['visualization_dir']) / settings['visualization_pattern'].format(epoch=epoch)),
        bbox_inches='tight'  # 防止边缘被裁剪
    )
    plt.close()  # 关闭图形避免内存泄漏


def save_loss_curve(train_loss_history, val_loss_history, settings):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(settings['loss_curve'])
    plt.close()
