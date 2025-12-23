import torch, timm
import torch.nn as nn
from timm.layers.squeeze_excite import EffectiveSEModule
from collections import OrderedDict
import torch.nn.functional as F

class RDNet_Tiny(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('rdnet_tiny.nv_in1k', pretrained=True, num_classes=0)
        self.fc = nn.Linear(1040, num_classes)

    def forward(self, x):
        embedings = self.backbone(x)
        out = self.fc(embedings)
        return embedings, out


class RDNet_Small(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('rdnet_small.nv_in1k', pretrained=True, num_classes=0)
        self.fc = nn.Linear(1264, num_classes)

    def forward(self, x):
        embedings = self.backbone(x)
        out = self.fc(embedings)
        return embedings, out


class RDNet_Base(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('rdnet_base.nv_in1k', pretrained=True, num_classes=0)
        self.fc = nn.Linear(1760, num_classes)
        # print(self.backbone)

    def forward(self, x):
        embedings = self.backbone(x)
        out = self.fc(embedings)
        return embedings, out


class RDNet_Large(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('rdnet_large.nv_in1k', pretrained=True, num_classes=0)
        self.fc = nn.Linear(2000, num_classes)

    def forward(self, x):
        embedings = self.backbone(x)
        out = self.fc(embedings)
        return embedings, out
    



# SpatialAttentionModule
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.latest_input_feature = None # 用於儲存最後的輸入特徵 (B, C, H, W)
        self.latest_attn_map = None  # 用於儲存最後的注意力圖 (B, 1, H, W)

    def forward(self, x):
        self.latest_input_feature = x.detach() # 獲取輸入特徵
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv1(x_out))
        self.latest_attn_map = attn.detach() # 注意力圖

        return x * attn


# --- 主模型包裝 (已修改為結構替換) ---
class RDNet_Base_SAttention(nn.Module):
    """
    包裝器類：
    - 載入 rdnet_base.nv_in1k(timm 預訓練模型）
    - 在每個 EffectiveSEModule 後插入 SpatialAttentionModule (使用 nn.Sequential 替換)
    """

    def __init__(self, num_classes: int, sa_kernel_size: int = 7, drop_rate=0.2):
        super().__init__()

        # 載入預訓練模型
        self.model = timm.create_model("rdnet_base.nv_in1k", pretrained=True, num_classes=0, drop_path_rate=drop_rate)
        # self.fc = nn.Linear(1760, num_classes)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # 新增 Dropout
            NormedLinear(1760, num_classes)
        )
        # 執行注入
        self._inject_spatial_attention(sa_kernel_size)

    # -------------------------------------------------
    def _inject_spatial_attention(self, kernel_size):
        """
        找出所有 EffectiveSEModule, 並使用 nn.Sequential 替換
        以包含原 ESE 和新的 SA 模組。
        """
        sa_index = 0
        
        # 使用 list() 複製，因為我們會在迭代時修改 self.model 結構
        for name, module in list(self.model.named_modules()):
            if isinstance(module, EffectiveSEModule):
                
                # 創建新的 SA 實例
                sa_instance = SpatialAttentionModule(kernel_size)
                
                # 定位父模組和子模組名稱
                # name = 'stages.0.blocks.0.ese' (timm model 的命名方式)
                parts = name.rsplit('.', 1)

                if len(parts) == 2:
                    parent_path, child_name = parts
                else:
                    # ESEModule 是頂層模組
                    parent_path = ''
                    child_name = parts[0]
                
                # 獲取父模組的引用
                if parent_path:
                    try:
                        # 嘗試使用 timm/PyTorch 的 get_submodule
                        parent_module = self.model.get_submodule(parent_path)
                    except AttributeError:
                        # 如果 get_submodule 不存在，使用 Python 標準方法:
                        parent_module = self.model
                        for part in parent_path.split('.'):
                            parent_module = getattr(parent_module, part)
                else:
                    parent_module = self.model

                # 進行結構替換, 使用 OrderedDict 構造 nn.Sequential 🌟
                new_sequence = nn.Sequential(OrderedDict([
                    ('original_ese', module),           # 原有的 ESE (nn.Module)
                    ('spatial_attn', sa_instance)       # 新增的 SA (nn.Module)
                ]))

                # 在父模組上執行替換 (parent_module.child_name = new_sequence)
                setattr(parent_module, child_name, new_sequence)
                
                print(f"✅ 已將 {name} 替換為 Sequential(ESE + SA)")
                sa_index += 1

        if sa_index == 0:
            print("⚠️ 未找到任何 EffectiveSEModule，請確認模型結構是否符合。")


    # -------------------------------------------------
    def forward(self, x):
        """整體前向傳遞: backbone (含 SA)、GAP、FC"""
        x = self.model.forward_features(x)
        # 這裡的 x 已經是 1760 維的向量
        x = self.model.head(x) 
        out = self.fc(x)
        return x, out

    # -------------------------------------------------
    # --- 參數分組 (已修改為匹配新的結構) ---
    def get_sa_parameters(self):
        """返回所有注入的 SA 模組參數 (通過結構名稱查找)"""
        sa_params = []
        for name, module in self.model.named_modules():
            # 我們尋找在結構替換中命名為 'spatial_attn' 的子模組
            if isinstance(module, SpatialAttentionModule) and name.endswith('.spatial_attn'):
                sa_params.extend(list(module.parameters()))
        return iter(sa_params)

    def get_head_parameters(self):
        """返回分類頭參數"""
        return self.fc.parameters()

    def get_backbone_parameters(self):
        """返回骨幹參數(排除 SA 與 Head)"""
        # 使用集合操作來排除是更穩健的方法
        all_params = set(self.model.parameters())
        sa_params = set(list(self.get_sa_parameters()))
        head_params = set(list(self.get_head_parameters()))
        
        # 由於 self.model.head(x) 包含了 pooling/norm，其參數已經包含在 self.model.parameters() 中。
        # 我們假設 head 的參數都是預訓練權重，應包含在 backbone_params 中。
        # 這裡保持與原始邏輯一致，僅排除 SA 參數：
        backbone_params = all_params - sa_params
        
        # 注意：如果 self.model.head 內部有可訓練的參數，它們會被包含在 backbone_params 中，
        # 並根據 freeze_and_unfreeze_params 的邏輯進行處理。
        
        return iter(backbone_params)

    # ------------------------------------------------
    def update_training_stage(self, stage=1):
        """
        根據訓練階段調整凍結策略。
        Stage 1: 凍結所有 Backbone, 只訓練 SA 和 FC。
        Stage 2: 保持 SA/FC 可訓練，並解凍第一層 ESE 之後的主幹網路。
        """
        if stage == 1:
            print(f"\n[Model] 切換至 Stage 1: 鎖定 Backbone, 只訓練 SA 和 FC")
            # 1. 凍結主幹網路的所有參數
            for param in self.model.parameters():
                param.requires_grad = False
            
            # 2. 解凍 SA 和 FC (永遠需要訓練)
            for param in self.fc.parameters():
                param.requires_grad = True
            for param in self.get_sa_parameters():
                param.requires_grad = True
                
        elif stage == 2:
            print(f"\n[Model] 切換至 Stage 2: 解凍部分 Backbone 進行微調")
            # 這裡不需重新鎖定，直接基於目前狀態去解凍特定層
            found_first_ese = False
            # 遍歷主幹網路模組，找到第一個 ESEModule 並開始解凍其後的參數
            for name, module in self.model.named_modules():
                # 識別第一個 ESEModule (即結構替換後的那個)
                if not found_first_ese and isinstance(module, EffectiveSEModule):
                    found_first_ese = True 
                    # 解凍
                    for param in module.parameters():
                        param.requires_grad = True
                # 解凍第一個 ESEModule 之後的所有層
                elif found_first_ese:
                    for param in module.parameters():
                        param.requires_grad = True
            
            print("- Stage 2 設定完成: SA, FC 及第一層 ESE 後的參數已解凍。")

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        # 權重形狀改為 (out_features, in_features) 符合 PyTorch Linear 標準
        # 這樣 F.linear 自動轉置後才會變成 (in, out)，才能跟輸入相乘
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # 初始化權重
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        # 對輸入特徵 x 做歸一化 (沿著 feature 維度)
        out = F.normalize(x, dim=1) 
        # 對權重做歸一化時，因為形狀變了，現在要沿著 dim=1 (feature 維度) 歸一化
        # 這樣每個類別的權重向量長度都會是 1
        normed_weight = F.normalize(self.weight, dim=1)
        # 3. 計算 Cosine Similarity (也就是歸一化後的 Linear)
        out = F.linear(out, normed_weight)
        return out


# 自定義偵錯模組
class DebugShape(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"[{self.name}] Input Shape: {x.shape}")
        return x
