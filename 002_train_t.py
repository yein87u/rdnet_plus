import datasets
import config
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader as DL
import timm
import utils
from tqdm import tqdm
import os
from models import rdnet_tiny, rdnet_small, rdnet_large, rdnet_base
from models import RDNet_Tiny, RDNet_Small, RDNet_Base, RDNet_Large, RDNet_Base_SAttention
import numpy as np

from typing import Dict, Any
import torch.nn as nn

from Loss import LDAMLoss, FocalLoss
from torch.utils.data import WeightedRandomSampler

def _GetModel(args):
    if args.modelName == "rdnet_small.nv_in1k":
        print("Use [rdnet_small.nv_in1k]")
        model = RDNet_Small(num_classes=args.classes)
    elif args.modelName == "rdnet_tiny.nv_in1k":
        print("Use [rdnet_tiny.nv_in1k]")
        model = RDNet_Tiny(num_classes=args.classes)
    elif args.modelName == "rdnet_base.nv_in1k":
        print("Use [rdnet_base.nv_in1k]")
        model = RDNet_Base(num_classes=args.classes)
    elif args.modelName == "rdnet_large.nv_in1k":
        print("Use [rdnet_large.nv_in1k]")
        model = RDNet_Large(num_classes=args.classes)
    elif args.modelName == 'rdnet_base_SAttention':
        print("Use [rdnet_base & spatial attention]")
        model = RDNet_Base_SAttention(num_classes=args.classes, sa_kernel_size=3, drop_rate=0.2)
    
    # model = timm.create_model(
    #     args.modelName, 
    #     pretrained=True, 
    #     in_chans=3, 
    #     num_classes=args.classes,
    # ).to(args.device)

    return model

def _GetOptimizer(args, model: nn.Module):
    if '.nv_in1k' in args.modelName:
        param_groups = model.parameters()
    else:
        optimizer = None
        # 定義學習率乘數, 新模組 (SA和FC) 相對於微調層的學習率乘數, 5 or 10
        LR_MULTIPLIER = 2.0

        # SA 模組參數
        sa_params = model.get_sa_parameters()
        # FC 層參數
        fc_params = model.fc.parameters()

        # 獲取所有可訓練參數的集合
        all_trainable_params = [p for p in model.parameters() if p.requires_grad]
        # 將 SA 和 FC 參數轉換為集合以便於排除
        sa_fc_set = set(list(model.fc.parameters()) + list(model.get_sa_parameters()))
        # 主幹網路微調參數 = 所有可訓練參數 - SA參數 - FC參數
        backbone_params = [p for p in all_trainable_params if p not in sa_fc_set]
        
        # 創建參數字典列表
        param_groups = [
            # Group 1: 主幹網路微調 (預訓練權重) - 使用基礎LR
            {'params': backbone_params, 'lr': args.lr, 'name': 'backbone_finetune'},
            # Group 2: FC 層和 SA 模組 (隨機初始化) - 使用較高LR
            {'params': list(sa_params) + list(fc_params), 'lr': args.lr * LR_MULTIPLIER, 'name': 'new_modules'},
        ]

    # --- 4. 初始化優化器 ---
    if(args.optimizer == "adamw"):
        optimizer = torch.optim.AdamW(
            param_groups,  # 傳入參數字典列表
            lr=args.lr,  # 這裡設置的lr會被param_groups中的lr覆蓋
            weight_decay=args.weight_decay
        )
    
    return optimizer

def _GetScheduler(args, optimizer):
    scheduler = None
    if(args.scheduler == "CosineAnnealingWarmRestarts"):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5, 
            T_mult=1, 
            eta_min=1e-6
        )
    if(args.scheduler == "CosineAnnealingLR"):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=50,  # 總訓練 Epoch 數
            eta_min=2e-5
        )
    
    return scheduler


def one_epoch(args, model, dataloader, optimizer, criterion):
    if args.phase == "train":
        model.train()
    elif args.phase == "val":
        model.eval()
    
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with tqdm(dataloader, desc=f"{args.phase}", ncols=100, leave=False) as pbar:
        for images, labels in pbar:
            # 若為train階段, 打開梯度追蹤
            with torch.set_grad_enabled(args.phase == "train"):
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                if '.nv_in1k'  in args.modelName or 'SAttention' in args.modelName:
                    features, logits = model(images)   #output.shape = (batch pred_result)
                else:
                    logits = model(images)   #output.shape = (batch pred_result)
                
                loss = criterion(logits, labels)

                if args.phase == "train":
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            
            pbar.set_postfix({
                    "Acc": f"{(correct/total*100):.2f}%",
                    "Loss": f"{(total_loss/total):.4f}"
                })

        epoch_loss = total_loss / total
        acc, f1, precision, recall = utils.evaluate(all_labels, all_preds)

    return acc, f1, precision, recall, epoch_loss


def _SaveModel(args, model):
    # 建立資料夾，例如 rdnet_small_fold1_v1_bz16
    folder_name = f"{args.modelName}_bz{args.batch_size}"
    path = os.path.join(args.root_model, folder_name + "_" + args.train_version)
    if not os.path.isdir(path):
        os.makedirs(path)

    # 儲存 checkpoint
    checkpoint_path = os.path.join(path, f"{args.modelName}_ckpt.pth.tar")
    utils.save_checkpoint(
        checkpoint_path,
        {
            'args': args,
            'Accuracy': args.best_acc,
            'Loss': args.best_loss,
            'F1-Score': args.best_f1,
            'Precision': args.best_precision,
            'Recall': args.best_recall,
            'model_state_dict': model.state_dict(),
        },
        False
    )

    print(" | ~~New Best Model Find!~~")

def main(args):
    print("===== main start =====")

    train_dataset = datasets.ImagesDataset(args=args, phase='train')
    print(f'mean: {args.mean}\nstd: {args.std}')
    print("Train dataset:", len(train_dataset))

    # === 讀取 Val Dataset ===
    val_dataset = datasets.ImagesDataset(args=args, phase='val')

    train_dataloader = DL(train_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    shuffle=True, 
                    drop_last=True)

    val_dataloader = DL(val_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.workers, 
                    shuffle=False)
    
    # 初始化每個 fold 的 metrics、model、optimizer、scheduler
    args = config.MetricsInit(args)
    model = _GetModel(args)

    base_lr = args.lr
    if hasattr(model, 'update_training_stage'):
        model.update_training_stage(stage=1)

    # print(model)
    # # 計算所有參數數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = _GetOptimizer(args, model)
    scheduler = _GetScheduler(args, optimizer)

    cls_num_list = train_dataset.cls_num_list
    criterion = nn.CrossEntropyLoss()
    # 備用 Loss： LDAM 帶有邊界調整
    ldam_criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.8, s=30) # max_m=0.5 和 s=30 是常見的起始值

    # 初始 Loss 函數設定為基準 Loss
    ldam_criterion = ldam_criterion.to(args.device)

    model, optimizer, train_dataloader, val_dataloader, \
    criterion, scheduler = args.accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader,
        criterion, scheduler
    )    
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }
    
    for epoch in range(args.epochs):
        args.epoch = epoch + 1

        if args.epoch == args.unfreeze_epoch and hasattr(model, 'update_training_stage'):
            print(f"\n🌟 Epoch {args.epoch}: 觸發解凍邏輯！重新初始化優化器...")
            # 解凍模型層
            model.update_training_stage(stage=2)
            
            # 調整學習率 (微調階段通常用更小的 LR)
            args.lr = base_lr * 0.5
            # 重新建立優化器, 因為參數的 requires_grad 變了，舊的 optimizer 不會更新新解凍的層
            optimizer = _GetOptimizer(args, model)
            # 重新建立 Scheduler, 因為 optimizer 換了
            scheduler = _GetScheduler(args, optimizer)
            # Accelerator必須重新 prepare 新的 optimizer
            if hasattr(args, 'accelerator'):
                optimizer, scheduler = args.accelerator.prepare(optimizer, scheduler)
                print("✨ Accelerator: Optimizer re-prepared.")

        if args.epoch == args.drw_start_epoch:
            print(f"\n⚡ Epoch {args.epoch}: 啟用 LDAM Loss (Deferred Re-Weighting)")
            # 將訓練使用的 Loss 函數切換為帶有邊界調整的 LDAM Loss
            criterion = ldam_criterion.to(args.device)
        
        # === Training ===
        args.phase = "train"
        train_acc, train_f1, train_precision, \
        train_recall, train_loss = one_epoch(args, 
                                            model, 
                                            train_dataloader, 
                                            optimizer, 
                                            criterion)
        
        # === Validation ===
        args.phase = "val"
        val_acc, val_f1, val_precision, \
        val_recall, val_loss = one_epoch(args, 
                                        model, 
                                        val_dataloader, 
                                        optimizer, 
                                        criterion)
        
        scheduler.step()
        args.lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch{args.epoch} T_Acc: {train_acc:.2f}% | V_Acc: {val_acc:.2f} | T_Loss: {train_loss:.4f} | V_Loss: {val_loss:.4f} | T_F1: {train_f1:.4f} | V_F1: {val_f1:.4f} | LR: {args.lr:.2e}", end="")
        
        # 儲存 best model
        if val_acc > args.best_acc:
            args.best_acc = val_acc
            args.best_loss = val_loss
            args.best_f1 = val_f1
            args.best_recall = val_recall
            args.best_precision = val_precision
            _SaveModel(args, model)
        else:
            print()
    
    utils.plot_history(history, args)

    print(f"!~~~~~~~~~~~~~~~!\nBest Acc = {args.best_acc:.2f}%\nBest Loss = {args.best_loss:.4f}\n!~~~~~~~~~~~~~~~!\n")
    print("===== main end =====")

if __name__ == '__main__':
    main(config.GetArgument())
