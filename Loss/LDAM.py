import torch
import torch.nn as nn
import numpy as np

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        
        # 確保 cls_num_list 是 numpy array
        cls_num_list = np.array(cls_num_list) 
        
        # 邊界 m_i 與 1/sqrt(n_i) 成正比
        # 這裡使用更平滑的 1/sqrt(n_i) 作為基礎，並進行正規化
        m_list = 1.0 / np.sqrt(cls_num_list)
        m_list = m_list * (max_m / np.max(m_list))
        
        # 將邊界值轉為 FloatTensor 並移至 GPU
        # self.m_list = torch.FloatTensor(m_list).cuda() 
        self.register_buffer('m_list', torch.FloatTensor(m_list))
        self.s = s # scaling factor
        
    def forward(self, x, target):
        """
        x: 模型輸出的 Logits (N, C)
        target: 樣本的真實標籤 (N)
        """
        # 1. 獲取 batch 中每個樣本對應的邊界值 m
        m = self.m_list[target]
        
        # 2. 構建一個 mask 來調整 Logits
        # index 用於標記正確類別的位置
        index = torch.zeros_like(x, dtype=torch.bool).to(x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        # 3. 調整 Logits：
        # 正確類別的 Logit 減去邊界 m (x - m)
        # 非正確類別的 Logit 保持不變 (x)
        # Logit 調整是通過將 x_m 放置到正確類別的位置實現的
        x_m = x - m.unsqueeze(1)
        x_adjusted = torch.where(index, x_m, x)
        
        # 4. 使用 scaling factor 's' 調整並計算標準交叉熵
        return nn.functional.cross_entropy(x_adjusted * self.s, target)