import torch
from torch.utils.data import DataLoader as DL
from tqdm import tqdm
import os
import datasets
import config
import utils
from models import rdnet_tiny, rdnet_small, rdnet_large, rdnet_base
from models import RDNet_Tiny, RDNet_Small, RDNet_Base, RDNet_Large, RDNet_Base_ComplexHead, RDNet_Base_SAttention
# 加速
from accelerate import Accelerator
import timm

from Loss import LDAMLoss, FocalLoss


def _GetModel(args, device):
    if args.modelName == "rdnet_small.nv_in1k":
        print("Use [rdnet_small.nv_in1k]")
        model = RDNet_Small(num_classes=args.classes).to(device)
    elif args.modelName == "rdnet_tiny.nv_in1k":
        print("Use [rdnet_tiny.nv_in1k]")
        model = RDNet_Tiny(num_classes=args.classes).to(device)
    elif args.modelName == "rdnet_base.nv_in1k":
        print("Use [rdnet_base.nv_in1k]")
        model = RDNet_Base(num_classes=args.classes).to(device)
    elif args.modelName == "rdnet_large.nv_in1k":
        print("Use [rdnet_large.nv_in1k]")
        model = RDNet_Large(num_classes=args.classes).to(device)
    elif args.modelName == 'rdnet_base_reload_head':
        print("Use [rdnet_base & reload_head]")
        model = RDNet_Base_ComplexHead(num_classes=args.classes).to(device)
    elif args.modelName == 'rdnet_base_SAttention':
        print("Use [rdnet_base & spatial attention]")
        model = RDNet_Base_SAttention(num_classes=args.classes, sa_kernel_size=3)
    
    # model = timm.create_model(
    #     args.modelName, 
    #     pretrained=True, 
    #     in_chans=3, 
    #     num_classes=args.classes,
    # ).to(args.device)
    
    return model


def load_best_model(args, model):
    """
    載入訓練好的最佳模型 checkpoint
    """
    ckpt_path = './checkpoint/rdnet_base_SAttention_bz16__v4/rdnet_base_SAttention_ckpt_epoch24.pth.tar'
    print(f"✅ 載入模型權重：{ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    args = checkpoint['args']
    return model, args


def test_one_epoch(args, model, dataloader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, correct, total = 0, 0, 0

    with tqdm(dataloader, desc="Testing", ncols=100) as pbar:
        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            with torch.no_grad():
                # if 'SAttention' in args.modelName:
                if '.nv_in1k'  in args.modelName or 'SAttention' in args.modelName:
                    features, logits = model(images)
                else:
                    logits = model(images)
                loss = criterion(logits, labels)
                _, preds = torch.max(logits, 1)

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                "Acc": f"{(correct/total*100):.2f}%",
                "Loss": f"{(total_loss/total):.4f}"
            })

    acc, f1, precision, recall = utils.evaluate(all_labels, all_preds)
    loss = total_loss / total

    return acc, f1, precision, recall, loss


def main(args):
    print("===== Test start =====")
    
    #建立加速器
    accelerator = Accelerator()
    device = accelerator.device

    # === 載入模型 ===
    model = _GetModel(args, device)
    model, args = load_best_model(args, model)

    # === 載入 Test Dataset ===
    test_dataset = datasets.ImagesDataset(args=args, phase='test')
    print("Test dataset:", len(test_dataset))

    test_dataloader = DL(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False
    )


    criterion = FocalLoss(
        gamma=2.0, # Gamma 越大，對多數類別的抑制越強，強制關注少數類別
        alpha=None,             # 先不使用 alpha，讓 gamma 專注於難易樣本分類
        reduction='mean', 
        task_type='multi-class',
        num_classes=args.classes
    )


    model, criterion = accelerator.prepare(model, criterion)
    print(model)

    # === 執行測試 ===
    acc, f1, precision, recall, loss = test_one_epoch(args, model, test_dataloader, criterion)

    print(f"\n===== Test Result =====")
    print(f"Accuracy : {acc:.2f}%")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Loss     : {loss:.4f}")
    print("========================\n")

    print("===== Test end =====")


if __name__ == '__main__':
    main(config.GetArgument())
