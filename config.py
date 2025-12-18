import torch
import random
import argparse
import numpy as np
# 加速
from accelerate import Accelerator

def _GetMainParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='Linux')
    parser.add_argument('--code_root', type=str, default='C:\\Users\\a1233\\Desktop\\rdnet_plus')
    parser.add_argument('--dataset_root', type=str, default='C:\\Users\\a1233\\Desktop\\Datasets\\PDD271')
    parser.add_argument('--pkl_path', type=str, default='./datasets/all_data.pkl')
    
    '''
    resnet50                            Total parameters:  24,063,311     test_acc: 84.11%    LR: 3e-5    weight_decay: 5e-3
    resnet101                           Total parameters:  43,055,439     test_acc: 82.59%    LR: 3e-5    weight_decay: 5e-4
    efficientnet_b0                     Total parameters:   4,354,699     test_acc: 86.67%    LR: 3e-5    weight_decay: 5e-4  
    efficientnet_b1                     Total parameters:   6,860,335     test_acc: 87.21%    LR: 3e-5    weight_decay: 5e-4
    efficientnet_b2                     Total parameters:   8,082,833     test_acc: 85.29%    LR: 3e-5    weight_decay: 5e-4
    mobilenetv1_100                     Total parameters:   3,484,751     test_acc: 84.75%    LR: 3e-5    weight_decay: 5e-4

    rdnet_base.nv_in1k_timm             Total parameters:  86,167,327     test_acc: 94.40%    LR: 3e-5    weight_decay: 1e-4

    rdnet_tiny.nv_in1k                  Total parameters:  23,104,119     test_acc: 92.53%    LR: 3e-5    weight_decay: 1e-4
    rdnet_small.nv_in1k                 Total parameters:  49,514,263     test_acc: 92.63%    LR: 3e-5    weight_decay: 1e-4
    rdnet_base.nv_in1k                  Total parameters:  86,167,327     test_acc: 94.10%    LR: 3e-5    weight_decay: 1e-4
    rdnet_large.nv_in1k                 Total parameters: 184,810,903     test_acc: 92.43%    LR: 3e-5    weight_decay: 1e-4

    rdnet_base_CLAHE_v1                 Total parameters:  86,167,327     test_acc: 94.59%    LR: 5e-5    weight_decay: 1e-4 (rdnet_base.nv_in1k)
    rdnet_base_CLAHE_v2                 Total parameters:  86,167,327     test_acc: 94.69%    LR: 5e-5    weight_decay: 1e-4 (rdnet_base.nv_in1k)

    rdnet_base_CLAHE_LDAM_v1            

    rdnet_base_CLAHE_reloadhead_v1      Total parameters:  85,941,375     test_acc: 94.59%    LR: 5e-5    weight_decay: 1e-4   (rdnet_base_reload_head)
    
    rdnet_base_CLAHE_SAttention_v1      Total parameters:  86,169,973     test_acc: 94.20     LR: 5e-5    weight_decay: 1e-4   (rdnet_base_SAttention)

    python train_t.py --modelName "rdnet_tiny.nv_in1k" --train_version "r_v1"
    python test_t.py --modelName "rdnet_large.nv_in1k"


    
    rdnet_base.nv_in1k (隨機切分)                   Total parameters: 86,167,327      val_acc: 91.44%    val_loss: 0.48     test_acc: 92.43%     test_loss: 0.46 (_origin_v2)
    
    rdnet_base.nv_in1k (按類別切分, LDAM)           Total parameters: 86,167,327      val_acc: 89.86%    val_loss: 5.86     test_acc: 90.46%     test_loss: 5.00 (_origin_v3)
    rdnet_base.nv_in1k (呈上, 資料增強)             Total parameters: 86,167,327      val_acc: 92.13%   val_loss: 4.51     test_acc: 91.74%     test_loss: 4.14       F1: 66.8927 (_origin_v4)
    rdnet_base.nv_in1k (呈上, bz:16)               Total parameters: 86,167,327      val_acc: 92.42%   val_loss: 4.07     test_acc: 93.31%     test_loss: 3.68       F1: 69.8019 (_origin_v5)
    rdnet_base.nv_in1k (呈上, 換成FocalLoss)       Total parameters: 86,167,327      val_acc: 93.01%   val_loss: 0.0008   test_acc: 93.31%     test_loss: 0.0006     F1: 71.7065 (_origin_v6)
    
    rdnet_base_SAttention(第一個ESE前凍結)              Total parameters: 81,719,381       val_acc: 93.70%   val_loss: 0.0008   test_acc: 94.69%     test_loss: 0.0005     F1: 76.9600 (SAttention_v1)
    rdnet_base_SAttention(呈上, 加入損失alpha)          Total parameters: 81,719,381       val_acc: 92.81%   val_loss: 0.0008   test_acc: 94.40%     test_loss: 0.0006     F1: 75.9733 (SAttention_v2)
    rdnet_base_SAttention(呈5, lr LR_MUL更改)           Total parameters: 81,719,381       val_acc: 93.50%   val_loss: 0.0008   test_acc: 94.20%     test_loss: 0.0006     F1: 74.6704 (SAttention_v3)
    rdnet_base_SAttention(呈5, ksize:7=>3)              Total parameters: 81,717,221        val_acc: 93.01   val_loss: 0.0009    test_acc: 94.99%     test_loss: 0.0006      F1: 78.0857 (SAttention_v4)
    rdnet_base_SAttention(呈7, 取代 CLAHE)   
    
    # 換資料集切分方法後 /new_checkpoints
    1.rdnet_base_SAttention                                 Total parameters: 81,717,221       test_acc: 92.50      F1: 71.50 (v1)
    2.rdnet_base_SAttention(1*1convfc, HSV, lightness)      Total parameters: 81,717,221       test_acc: 92.41%     F1: 71.42 (v2)
    3.rdnet_base_SAttention(呈1, train加入sampler)          Total parameters: 81,717,221        test_acc: 91.42%     F1: 68.32 (v3)
    4.呈1 (換成LDAM+DRW+CHALE, 分類頭正規化,drop)           Total parameters: 477,446 => 81,717,221   test_acc: 93.29%    F1: 74.05 (v5)
    '''
    parser.add_argument('--modelName', type=str, default="rdnet_base_SAttention")

    parser.add_argument('--optimizer', type=str, default="adamw")
    parser.add_argument('--scheduler', type=str, default="CosineAnnealingWarmRestarts") # CosineAnnealingWarmRestarts
    
    parser.add_argument('--phase', type=str, default="train")
    parser.add_argument('--train_version', type=str, default="v6")
    parser.add_argument('--pretrained_path', type=str, default="./weight")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--classes', type=int, default=271)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5) # 3e-5、1e-3
    parser.add_argument('--weight_decay', type=float, default=1e-2) # 5e-3、1e-4

    parser.add_argument('--image_size', type=list, default=[224, 224])
    parser.add_argument('--seed', type=int, default=42)

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


