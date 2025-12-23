import torch
import random
import argparse
import numpy as np
# 加速
from accelerate import Accelerator


'''
resnet50                            Total parameters:  24,063,311     test_acc: 84.11%    LR: 3e-5    weight_decay: 5e-3
resnet101                           Total parameters:  43,055,439     test_acc: 82.59%    LR: 3e-5    weight_decay: 5e-4
efficientnet_b0                     Total parameters:   4,354,699     test_acc: 86.67%    LR: 3e-5    weight_decay: 5e-4  
efficientnet_b1                     Total parameters:   6,860,335     test_acc: 87.21%    LR: 3e-5    weight_decay: 5e-4
efficientnet_b2                     Total parameters:   8,082,833     test_acc: 85.29%    LR: 3e-5    weight_decay: 5e-4
mobilenetv1_100                     Total parameters:   3,484,751     test_acc: 84.75%    LR: 3e-5    weight_decay: 5e-4

rdnet_base_SAttention(換成LDAM+DRW+CHALE, 分類頭正規化,dropout)           Total parameters: 477,446 => 81,717,221   test_acc: 93.29%    F1: 74.05 (final)


整體影響
rdnet_base.nv_in1k(timm)                             TOP 1: 90.68   TOP 5: 97.34
rdnet_base_SAttention(final)                         TOP 1: 93.29   TOP 5: 98.42

其他策略保留一樣
rdnet_base_SAttention(noSA)                          TOP 1: 91.17   TOP 5: 98.03
rdnet_base_SAttention                                TOP 1: 93.29   TOP 5: 98.42

(LDAM+DRW)
rdnet_base_SAttention(origin_fc)                      TOP 1: 91.57   TOP 5: 98.03
rdnet_base_SAttention(drop, normlinear)               TOP 1: 93.29   TOP 5: 98.42

(LDAM+DRW)
rdnet_base_SAttention(noDA_v1)                        TOP 1: 91.47   TOP 5: 98.13
rdnet_base_SAttention(翻轉、旋轉, exp_origin_v1)       TOP 1: 93.20   TOP 5: 97.83
rdnet_base_SAttention(HSV_brightness_v1)              TOP 1: 92.80   TOP 5: 98.37
rdnet_base_SAttention(CHALE, final)                   TOP 1: 93.29   TOP 5: 98.42
'''

def _GetMainParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='Linux')
    parser.add_argument('--code_root', type=str, default='C:\\Users\\a1233\\Desktop\\rdnet_plus')
    parser.add_argument('--dataset_root', type=str, default='C:\\Users\\a1233\\Desktop\\Datasets\\PDD271')
    parser.add_argument('--pkl_path', type=str, default='./datasets/all_data.pkl')
    
    parser.add_argument('--modelName', type=str, default="rdnet_base_SAttention")

    parser.add_argument('--optimizer', type=str, default="adamw")
    parser.add_argument('--scheduler', type=str, default="CosineAnnealingWarmRestarts") # CosineAnnealingWarmRestarts
    
    parser.add_argument('--phase', type=str, default="train")
    parser.add_argument('--train_version', type=str, default="noSA_v1")
    parser.add_argument('--pretrained_path', type=str, default="./weight")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--classes', type=int, default=271)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5) # 3e-5、1e-3
    parser.add_argument('--weight_decay', type=float, default=1e-2) # 5e-3、1e-4

    parser.add_argument('--image_size', type=list, default=[224, 224])
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--unfreeze_epoch', type=int, default=20)
    parser.add_argument('--drw_start_epoch', type=int, default=35)

    parser.add_argument('--root_model', type=str, default='new_checkpoint')

    return parser.parse_args()

def MetricsInit(args):
    # 建立儲存最佳acc的變數
    args.best_loss = float('inf')
    args.best_acc = float('-inf')

    args.best_f1 = float('-inf')
    args.best_recall = float('-inf')
    args.best_precision = float('-inf')

    return args

def PathTranslate(args):
    if(args.system == 'Linux'):
        args.code_root = "/mnt/" + args.code_root[0].lower() + args.code_root[2:].replace("\\", "/")
        args.dataset_root = "/mnt/" + args.dataset_root[0].lower() + args.dataset_root[2:].replace("\\", "/")

    return args



def GetArgument():
    import warnings 
    warnings.filterwarnings("ignore")

    args = _GetMainParameters()

    args = MetricsInit(args)
    args = PathTranslate(args)
    
    #建立加速器
    args.accelerator = Accelerator()
    args.device = args.accelerator.device
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args


