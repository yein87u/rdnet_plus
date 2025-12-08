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
from models import RDNet_Base_SAttention, SpatialAttentionModule

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
        model = RDNet_Base_ComplexHead(num_classes=args.classes)
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

    # === 執行 SA 視覺化 (僅抓取第一個 Batch 的第一張影像) ===
    if args.modelName == 'rdnet_base_SAttention':
        VISUALIZATION_OUTPUT_DIR = "./sa_attention_maps"
        # 重新初始化 Dataloader 以確保拿到新的迭代器 (或確保在 test_one_epoch 後 Dataloader 仍可用)
        test_dataloader_viz = DL(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
        
        visualize_single_image_attention(args, model, test_dataloader_viz, VISUALIZATION_OUTPUT_DIR, device)


    print("===== Test end =====")


def visualize_single_image_attention(args, model, dataloader, output_dir, device):
    """
    從 dataloader 提取第一個 Batch 的所有影像，為每張影像創建獨立資料夾，
    並將注意力圖疊加到原始圖像上儲存。
    """
    # 🌟 修正點 1: 解包模型 🌟
    # model.module 用於 DistributedDataParallel/DataParallel，否則使用原模型
    model_unwrapped = model.module if hasattr(model, 'module') else model

    # 檢查模型是否包含 SA 模組
    if 'SAttention' not in model_unwrapped.__class__.__name__:
        print("⚠️ 模型不包含 SAttention 模組，跳過視覺化。")
        return

    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 獲取所有 SpatialAttentionModule 實例
    sa_modules = []
    # 🌟 修正點 2: 使用解包後的模型遍歷模組 🌟
    for name, module in model_unwrapped.named_modules():
        if isinstance(module, SpatialAttentionModule) and name.endswith('.spatial_attn'):
            sa_modules.append((name, module))
            
    if not sa_modules:
        print("模型中未找到 SpatialAttentionModule。")
        return

    print(f"\n找到 {len(sa_modules)} 個 SA 模組進行視覺化。")
    
    # 2. 只從 dataloader 中取出第一個 Batch
    try:
        first_batch = next(iter(dataloader)) 
        images, labels = first_batch
    except StopIteration:
        print("Dataloader 為空。")
        return
        
    # 3. 執行模型前向傳遞
    images_gpu = images.to(device)
    
    with torch.no_grad():
        # 運行模型，讓 SA 模組內儲存最新的 attn_map
        model(images_gpu) 

    # 4. 迭代整個 Batch 中的每張影像，並儲存結果
    batch_size = images.shape[0]
    total_saved_images = 0
    
    for batch_idx in range(batch_size):
        
        # A. 創建每個影像的獨立輸出資料夾
        image_output_dir = os.path.join(output_dir, f"image_{batch_idx:03d}")
        os.makedirs(image_output_dir, exist_ok=True)
        
        # B. 對當前影像進行反正規化 (使用 CPU 上的 images)
        single_image_denorm = utils.denormalize_image(images[batch_idx], args.mean, args.std) 
        
        # C. 迭代所有 SA 模組並儲存結果
        for i, (name, sa_module) in enumerate(sa_modules):
            
            # 從解包後的 SA 實例訪問儲存的注意力圖
            attn_map = sa_module.latest_attn_map
            if attn_map is None:
                continue
            
            # 提取當前 Batch 索引 (batch_idx) 的 Attention Map (H, W)
            single_map_np = attn_map[batch_idx, 0, :, :].cpu().numpy()
            
            # 獲取 blocks.X
            layer_name = name.rsplit('.', 2)[-2]
            
            # 儲存疊加後的圖像到該影像專屬資料夾
            file_name_overlay = f"SA_Overlay_Layer{i}_{layer_name}.png"
            overlay_output_path = os.path.join(image_output_dir, file_name_overlay)
            
            # 調用疊加函式
            utils.overlay_attention_on_image(
                original_img_tensor=single_image_denorm, 
                attn_map_np=single_map_np,
                output_path=overlay_output_path,
                layer_name=layer_name
            )
            total_saved_images += 1
            
    print(f"\n✨ 空間注意力疊加圖像已儲存至：{output_dir} (共 {batch_size} 個影像資料夾，{total_saved_images} 張圖片)")


if __name__ == '__main__':
    main(config.GetArgument())
