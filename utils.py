import os, gc, torch, shutil
import numpy as np
import sklearn.metrics as skm 
from PIL import Image
import torchvision
import matplotlib.pyplot as plt


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(filename, state, is_best):
    torch.save(state, filename)
    if is_best:
        print('Saving best model...')
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def print_model(args, model):
    from pytorch_model_summary import summary
    input1 = torch.zeros((1, 3, args.image_size[0], args.image_size[1]), device=args.device)
    print(summary(model, input1, show_input=False))
    print(args)


def flash():
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(AllTargets, AllPreds):
    acc = skm.accuracy_score(AllTargets, AllPreds)*100
    f1 = skm.f1_score(AllTargets, AllPreds, average='macro')*100
    precision = skm.precision_score(AllTargets, AllPreds, average='macro')*100
    recall = skm.recall_score(AllTargets, AllPreds, average='macro')*100    
    return acc, f1, precision, recall


def ComputingEER(label, pred, positive_label=1):
    # Ref：https://github.com/YuanGongND/python-compute-eer
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    # 將label：1視為positive

    fpr, tpr, thresholds = skm.roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr
    # find indices where EER, fpr100, fpr1000, fpr0, best acc occur
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    fpr100_idx = sum(fpr <= 0.01) - 1
    fpr1000_idx = sum(fpr <= 0.001) - 1
    fpr10000_idx = sum(fpr <= 0.0001) - 1
    fpr0_idx = sum(fpr <= 0.0) - 1

    # compute EER, FRR@FAR=0.01, FRR@FAR=0.001, FRR@FAR=0
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    fpr100 = fnr[fpr100_idx]
    fpr1000 = fnr[fpr1000_idx]
    fpr10000 = fnr[fpr10000_idx]
    fpr0 = fnr[fpr0_idx]

    metrics = (eer, fpr100, fpr1000, fpr10000, fpr0)
    metrics_thred = (thresholds[eer_idx], thresholds[fpr100_idx], thresholds[fpr1000_idx], thresholds[fpr10000_idx], thresholds[fpr0_idx])
    # print('EER:%.2f%%, FRR@FAR=0.01: %.2f%%, FRR@FAR=0.001: %.2f%%, FRR@FAR=0.0001: %.2f%%, FRR@FAR=0: %.2f%%, Aver: %.2f%%' %
        #   (eer * 100, fpr100 * 100, fpr1000 * 100, fpr10000 * 100, fpr0 * 100, np.mean(metrics) * 100))
    return eer


def plot_history(history, args):
    """根據訓練歷史數據繪製 Loss 和 Accuracy 曲線圖。"""
    path = args.code_root
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # --- 繪製 Loss 曲線圖 ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # 儲存 Loss 圖
    loss_path = os.path.join(path, f"loss_plot.png")
    plt.savefig(loss_path)
    plt.close() # 關閉當前圖形，防止記憶體洩漏
    
    # --- 繪製 Accuracy 曲線圖 ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    # 儲存 Acc 圖
    acc_path = os.path.join(path, f"acc_plot.png")
    plt.savefig(acc_path)
    plt.close() # 關閉當前圖形

    print(f"\n[圖表儲存成功] Loss 圖: {loss_path} | Acc 圖: {acc_path}")


# def save_attention_map_png(attn_map_np, output_path, layer_name):
#     """
#     將注意力圖 (NumPy array) 儲存為彩色熱力圖 PNG。
#     """
#     # 正規化到 0-255 範圍 (用於 PNG 輸出)
#     single_map = (attn_map_np - attn_map_np.min()) / (attn_map_np.max() - attn_map_np.min() + 1e-8)
#     single_map = (single_map * 255).astype(np.uint8)
    
#     # 使用 Matplotlib 儲存為熱力圖 PNG
#     plt.imshow(single_map, cmap='viridis')
#     plt.title(f"Layer: {layer_name}", fontsize=8)
#     plt.axis('off')
    
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#     plt.close()
import cv2

def denormalize_image(tensor, IMAGENET_MEAN, IMAGENET_STD):
    """
    將標準化後的 PyTorch Tensor (C, H, W) 反正規化到 [0, 1] 範圍。
    """
    # 這裡假設您的正規化參數是 Imagenet 的標準值
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
    
    # 反正規化
    denorm_tensor = tensor * std + mean
    
    # 裁剪到 [0, 1] 範圍並轉為 CPU
    return torch.clamp(denorm_tensor, 0, 1).cpu()

def overlay_attention_on_image(original_img_tensor, attn_map_np, output_path, layer_name):
    """
    將注意力熱力圖疊加到原始圖像上並儲存。
    
    Args:
        original_img_tensor (Tensor): 反正規化後的單張圖像 Tensor (3, H, W)。
        attn_map_np (numpy.ndarray): 0-1 範圍的注意力圖 (H, W)。
    """
    # 1. 將 Tensor 轉為 BGR 格式的 NumPy 圖像 (OpenCV 標準)
    # (H, W, 3) 範圍 [0, 255]
    img_np = original_img_tensor.permute(1, 2, 0).numpy()
    img_bgr = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR) # Matplotlib/PIL 是 RGB，OpenCV 是 BGR

    # 2. 將 Attention Map 轉為彩色熱力圖
    # 將 attn_map_np 縮放回原始圖像大小（因為 SA 輸出可能較小）
    attn_resized = cv2.resize(attn_map_np, (img_bgr.shape[1], img_bgr.shape[0]))
    
    # 顏色映射：使用 cv2.COLORMAP_JET 或 'viridis'
    heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 3. 執行疊加 (使用透明度 alpha)
    alpha = 0.5 # 熱力圖的透明度
    overlay = img_bgr.copy()
    cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0, overlay)
    
    # 4. 儲存結果 PNG
    cv2.imwrite(output_path, overlay)
    print(f"   [Overlay] Saved: {output_path}")