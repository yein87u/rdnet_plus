"""
RDNet
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
"""

from functools import partial
from typing import List

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.squeeze_excite import EffectiveSEModule
from timm.models import register_model, build_model_with_cfg, named_apply, generate_default_cfgs
from timm.models.layers import LayerNorm2d
from timm.layers import DropPath
__all__ = ["RDNet"]


# class RDNetClassifierHead(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         num_classes: int,
#         drop_rate: float = 0.,
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.num_features = in_features

#         self.norm = nn.LayerNorm(in_features)
#         self.drop = nn.Dropout(drop_rate)
#         self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

#     def reset(self, num_classes):
#         self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

#     def forward(self, x, pre_logits: bool = False):
#         x = x.mean([-2, -1])
#         x = self.norm(x)
#         x = self.drop(x)
#         if pre_logits:
#             return x
#         x = self.fc(x)
#         return x
    

# ===== Classifier Head =====-
class RDNetClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes, drop_rate=0.):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes)

    def reset(self, num_classes):
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward(self, x):
        x = x.mean([-2, -1])  # global average pool
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


# class PatchifyStem(nn.Module):
#     def __init__(self, num_input_channels, num_init_features, patch_size=4):
#         super().__init__()

#         self.stem = nn.Sequential(
#             nn.Conv2d(num_input_channels, num_init_features, kernel_size=patch_size, stride=patch_size),
#             LayerNorm2d(num_init_features),
#         )

#     def forward(self, x):
#         return self.stem(x)
    
# ===== Patchify Stem =====-
class PatchifyStem(nn.Module):
    # 為了切分patch設計的init
    def __init__(self, num_input_channels, num_init_features, patch_size=4):
    # def __init__(self, num_input_channels, num_init_features, ksize=3, stride=1):
        super().__init__()
        self.stem = nn.Sequential(
            # 為了切分patch設計的conv
            nn.Conv2d(num_input_channels, num_init_features,
                    kernel_size=patch_size, stride=patch_size),
            
            # 打破切分機制
            # nn.Conv2d(num_input_channels, num_init_features,
            #         kernel_size=ksize, stride=stride),
            LayerNorm2d(num_init_features)
        )

    def forward(self, x):
        return self.stem(x)


# class Block(nn.Module):
#     """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
#     def __init__(self, in_chs, inter_chs, out_chs):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),
#             LayerNorm2d(in_chs, eps=1e-6),
#             nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
#             nn.GELU(),
#             nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
#         )

#     def forward(self, x):
#         return self.layers(x)


# ===== Block & BlockESE =====
'''block_cls(num_input_features, inter_chs, growth_rate)'''
'''
Block chs:  120 480 96
Block chs:  216 864 96
Block chs:  312 1248 96

Block chs:  200 800 128
Block chs:  328 1312 128
Block chs:  456 1824 128
'''

class Block(nn.Module):
    """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
    def __init__(self, in_chs, inter_chs, out_chs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),   # DW Conv，提取空間特徵
            LayerNorm2d(in_chs, eps=1e-6),  # 計算每通道H*W通道平均值(Mean)和方差(Variance), 將輸出分布限制在一個穩定範圍, 有效解決梯度消失/爆炸問題
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),   # 擴展通道，增加特徵多樣性，H*W共享一個FC權重
            nn.GELU(),  # 引入非線性
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),  # 壓縮通道
        )

    def forward(self, x):
        return self.layers(x)


# class BlockESE(nn.Module):
#     """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
#     def __init__(self, in_chs, inter_chs, out_chs):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),
#             LayerNorm2d(in_chs, eps=1e-6),
#             nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
#             nn.GELU(),
#             nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
#             EffectiveSEModule(out_chs),
#         )

#     def forward(self, x):
#         return self.layers(x)


class BlockESE(nn.Module):
    """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
    def __init__(self, in_chs, inter_chs, out_chs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),   # DW Conv，提取空間特徵
            LayerNorm2d(in_chs, eps=1e-6),  # 計算每通道H*W通道平均值(Mean)和方差(Variance)，將輸出分布限制在一個穩定範圍有效解決梯度消失/爆炸問題
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),   # 擴展通道，增加特徵多樣性，H*W共享一個FC權重
            nn.GELU(),  # 引入非線性
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),  # 壓縮通道
            EffectiveSEModule(out_chs),
        )
        '''
            EffectiveSEModule:[
            GAP, # 已經捕捉H*W空間維度上的
            Conv(C, C/r, ksize = 1), # 從C個通道中計算C/r個精華訊息, 精煉特徵
            ReLU, # 引入非線性
            Conv(C/r, C, ksize = 1), # 透過可學習的權重, 從精華訊息中映射出C個預測權重分數
            Sigmoid # 正規化到區間[0, 1]，獲得通道權重向量
            ]
        '''

    def forward(self, x):
        return self.layers(x)
    
''''
class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False, gate_layer='hard_sigmoid', **_):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)
'''

# class DenseBlock(nn.Module):
#     def __init__(
#         self,
#         num_input_features,
#         growth_rate,
#         bottleneck_width_ratio,
#         drop_path_rate,
#         drop_rate=0.0,
#         rand_gather_step_prob=0.0,
#         block_idx=0,
#         block_type="Block",
#         ls_init_value=1e-6,
#         **kwargs,
#     ):
#         super().__init__()
#         self.drop_rate = drop_rate
#         self.drop_path_rate = drop_path_rate
#         self.rand_gather_step_prob = rand_gather_step_prob
#         self.block_idx = block_idx
#         self.growth_rate = growth_rate

#         self.gamma = nn.Parameter(ls_init_value * torch.ones(growth_rate)) if ls_init_value > 0 else None
#         growth_rate = int(growth_rate)
#         inter_chs = int(num_input_features * bottleneck_width_ratio / 8) * 8

#         if self.drop_path_rate > 0:
#             self.drop_path = DropPath(drop_path_rate)

#         self.layers = eval(block_type)(
#             in_chs=num_input_features,
#             inter_chs=inter_chs,
#             out_chs=growth_rate,
#         )

#     def forward(self, x):
#         if isinstance(x, List):
#             x = torch.cat(x, 1)
#         x = self.layers(x)

#         if self.gamma is not None:
#             x = x.mul(self.gamma.reshape(1, -1, 1, 1))

#         if self.drop_path_rate > 0 and self.training:
#             x = self.drop_path(x)
#         return x

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv1(x_out))
        return x * attn

# ===== DenseBlock =====
class DenseBlock(nn.Module):
    def __init__(self, num_input_features, growth_rate, bottleneck_width_ratio, drop_path_rate, block_type="Block", **kwags):
        super().__init__()
        inter_chs = int(num_input_features * bottleneck_width_ratio / 8) * 8
        block_cls = Block if block_type == "Block" else BlockESE
        self.layers = block_cls(num_input_features, inter_chs, growth_rate)

        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.ones(growth_rate) * 1e-6)

        # 新增空間注意力機制
        self.spatial_attn = SpatialAttentionModule(kernel_size=7) if block_type == "BlockESE" else None

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.layers(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        # 應用 SpatialAttentionModule
        # if self.spatial_attn is not None:
        #     x = self.spatial_attn(x) # 維度不變: [B, growth_rate, H, W]

        if self.drop_path_rate > 0 and self.training:# 根據當時狀態設置
            x = self.drop_path(x)
        return x


# class DenseStage(nn.Sequential):
#     def __init__(self, num_block, num_input_features, drop_path_rates, growth_rate, **kwargs):
#         super().__init__()
#         for i in range(num_block):
#             layer = DenseBlock(
#                 num_input_features=num_input_features,
#                 growth_rate=growth_rate,
#                 drop_path_rate=drop_path_rates[i],
#                 block_idx=i,
#                 **kwargs,
#             )
#             num_input_features += growth_rate
#             self.add_module(f"dense_block{i}", layer)
#         self.num_out_features = num_input_features

#     def forward(self, init_feature):
#         features = [init_feature]
#         for module in self:
#             new_feature = module(features)
#             features.append(new_feature)
#         return torch.cat(features, 1)
    

# ===== DenseStage =====
class DenseStage(nn.Module):
    def __init__(self, num_blocks, num_input_features, drop_path_rate, growth_rate, **kwargs):
        super().__init__()
        layers = []
        for i in range(num_blocks):
            block = DenseBlock(num_input_features, growth_rate,
                            drop_path_rate=drop_path_rate[i], **kwargs)
            num_input_features += growth_rate
            layers.append(block)
        self.blocks = nn.ModuleList(layers)
        self.num_out_features = num_input_features

    def forward(self, x):
        feats = [x]
        for blk in self.blocks:
            new_feat = blk(feats)
            feats.append(new_feat)
        # 殘差連接(特徵圖直接相接)
        return torch.cat(feats, 1)


# class RDNet(nn.Module):
#     def __init__(
#         self,
#         num_init_features=64,
#         growth_rates=(64, 104, 128, 128, 128, 128, 224),
#         num_blocks_list=(3, 3, 3, 3, 3, 3, 3),
#         bottleneck_width_ratio=4,   
#         zero_head=False,    # -
#         in_chans=3,  # timm option [--in-chans]  +
#         num_classes=1000,  # timm option [--num-classes]
#         drop_rate=0.0,  # timm option [--drop: dropout ratio]+
#         drop_path_rate=0.0,  # timm option [--drop-path: drop-path ratio]
#         checkpoint_path=None,  # timm option [--initial-checkpoint] -
#         transition_compression_ratio=0.5,   # +
#         ls_init_value=1e-6, # +
#         is_downsample_block=(None, True, True, False, False, False, True),
#         block_type="Block",
#         head_init_scale: float = 1.,    # +
#         **kwargs,
#     ):
#         super().__init__()
#         assert len(growth_rates) == len(num_blocks_list) == len(is_downsample_block)

#         # ?
#         self.num_classes = num_classes
#         if isinstance(block_type, str):
#             block_type = [block_type] * len(growth_rates)

#         # stem
#         self.stem = PatchifyStem(in_chans, num_init_features, patch_size=4)

#         # features
#         self.feature_info = []
#         self.num_stages = len(growth_rates) # -
#         curr_stride = 4  # stem_stride  ?
#         num_features = num_init_features
#         dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(num_blocks_list)).split(num_blocks_list)]

#         dense_stages = []
#         for i in range(self.num_stages):
#             dense_stage_layers = []
#             if i != 0:
#                 compressed_num_features = int(num_features * transition_compression_ratio / 8) * 8
                
#                 # 替代
#                 k_size = stride = 1
#                 if is_downsample_block[i]:
#                     curr_stride *= 2
#                     k_size = stride = 2
                
#                 dense_stage_layers.append(LayerNorm2d(num_features))
#                 dense_stage_layers.append(
#                     nn.Conv2d(num_features, compressed_num_features, kernel_size=k_size, stride=stride, padding=0)
#                 )
#                 num_features = compressed_num_features

#             stage = DenseStage(
#                 num_block=num_blocks_list[i],
#                 num_input_features=num_features,
#                 growth_rate=growth_rates[i],
#                 bottleneck_width_ratio=bottleneck_width_ratio,
#                 drop_rate=drop_rate,
#                 drop_path_rates=dp_rates[i],
#                 ls_init_value=ls_init_value,
#                 block_type=block_type[i],
#             )
#             dense_stage_layers.append(stage)
#             num_features += num_blocks_list[i] * growth_rates[i]

#             if i + 1 == self.num_stages or (i + 1 != self.num_stages and is_downsample_block[i + 1]):
#                 self.feature_info += [
#                     dict(
#                         num_chs=num_features,
#                         reduction=curr_stride,
#                         module=f'dense_stages.{i}',
#                         growth_rate=growth_rates[i],
#                     )
#                 ]
#             dense_stages.append(nn.Sequential(*dense_stage_layers))
#         self.dense_stages = nn.Sequential(*dense_stages)

#         # classifier
#         self.head = RDNetClassifierHead(num_features, num_classes, drop_rate=drop_rate)

#         # initialize weights
#         named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

#         if zero_head:
#             nn.init.zeros_(self.head[-1].weight.data)
#             if self.head[-1].bias is not None:
#                 nn.init.zeros_(self.head[-1].bias.data)

#         if checkpoint_path is not None:
#             self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

#     @torch.jit.ignore
#     def get_classifier(self):
#         return self.head.fc

#     def reset_classifier(self, num_classes=0, global_pool=None):
#         assert global_pool is None
#         self.head.reset(num_classes)

#     def forward_head(self, x, pre_logits: bool = False):
#         return self.head(x, pre_logits=True) if pre_logits else self.head(x)

#     def forward_features(self, x):
#         x = self.stem(x)
#         x = self.dense_stages(x)
#         return x

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x

#     def group_matcher(self, coarse=False):
#         assert not coarse
#         return dict(
#             stem=r'^stem',
#             blocks=r'^dense_stages\.(\d+)',
#         )
    


# ===== RDNet 主體 =====
class RDNet(nn.Module):
    def __init__(self,
                num_classes=1000,
                num_init_features=64,
                growth_rates=(64, 104, 128, 128, 128, 128, 224),
                num_blocks_list=(3, 3, 3, 3, 3, 3, 3),
                bottleneck_width_ratio=4,
                drop_rate=0.0,
                drop_path_rate=0.0,
                is_downsample_block=(None, True, True, False, False, False, True),
                block_type="Block",
                in_chans=3,
                transition_compression_ratio=0.5,
                ls_init_value=1e-6,
                head_init_scale: float = 1.,):
        super().__init__()
        assert len(growth_rates) == len(num_blocks_list) == len(is_downsample_block)
        
        self.num_stages = len(growth_rates)

        self.num_classes = num_classes
        if isinstance(block_type, str):
            block_type = [block_type] * self.num_stages

        # stem
        self.stem = PatchifyStem(in_chans, num_init_features, patch_size=4)

        # features
        num_features = num_init_features
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(num_blocks_list)).split(num_blocks_list)]

        dense_stages = []
        for i in range(self.num_stages):
            dense_stage_layers = []
            if i != 0:
                compressed_num_features = int(num_features * transition_compression_ratio / 8) * 8
                
                stride = 1
                if is_downsample_block[i]:
                    stride = 2
                
                dense_stage_layers += [
                    LayerNorm2d(num_features),
                    nn.Conv2d(num_features, compressed_num_features, kernel_size=stride, stride=stride)
                ]
                num_features = compressed_num_features

            stage = DenseStage(
                num_blocks=num_blocks_list[i],
                num_input_features=num_features,
                growth_rate=growth_rates[i],
                drop_path_rate=dp_rates[i],
                bottleneck_width_ratio=bottleneck_width_ratio,
                drop_rate=drop_rate,
                ls_init_value=ls_init_value,
                block_type=block_type[i],
            )
            dense_stage_layers.append(stage)
            num_features += num_blocks_list[i] * growth_rates[i]
            dense_stages.append(nn.Sequential(*dense_stage_layers))

        self.dense_stages = nn.Sequential(*dense_stages)
        # classifier
        self.head = RDNetClassifierHead(num_features, num_classes, drop_rate=drop_rate)
        # self.head = self.create_custom_head(num_features, num_classes)
        # self.head = nn.Linear(1760, num_classes)

        # initialize weights
        self.apply(self._init_weights)
    
    # add !!
    @staticmethod
    def create_custom_head(in_features: int, num_classes: int):
        """定義您圖示中的複雜分類頭部"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1), 
            # nn.Dropout(0.3),  # 保持您的 Dropout
            nn.Linear(in_features, num_classes),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.dense_stages(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def rdnet_tiny(num_classes=1000, **kwargs):
    n_layer = 7
    return RDNet(
        num_classes=num_classes,
        num_init_features=64,
        growth_rates=[64] + [104] + [128] * 4 + [224],
        num_blocks_list=[3] * n_layer,
        is_downsample_block=(None, True, True, False, False, False, True),
        transition_compression_ratio=0.5,
        block_type=["Block"] + ["Block"] + ["BlockESE"] * 4 + ["BlockESE"],
        **kwargs
    )


# 不同大小模型
def rdnet_small(num_classes=1000, **kwargs):
    n_layer = 11
    return RDNet(
        num_classes=num_classes,
        num_init_features=72,
        growth_rates=[64] + [128] + [128] * (n_layer - 4) + [240] * 2,
        num_blocks_list=[3] * n_layer,
        is_downsample_block=(None, True, True, False, False, False, False, False, False, True, False),
        transition_compression_ratio=0.5,
        block_type=["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2,
        **kwargs
    )


def rdnet_base(num_classes=1000, **kwargs):
    n_layer = 11
    return RDNet(
        num_classes=num_classes,
        num_init_features=120,
        growth_rates=[96] + [128] + [168] * (n_layer - 4) + [336] * 2,
        num_blocks_list=[3] * n_layer,
        is_downsample_block=(None, True, True, False, False, False, False, False, False, True, False),
        transition_compression_ratio=0.5,
        block_type=["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2,
        **kwargs
    )


def rdnet_large(num_classes=1000, **kwargs):
    n_layer = 12
    return RDNet(
        num_classes=num_classes,
        num_init_features=144,
        growth_rates=[128] + [192] + [256] * (n_layer - 4) + [360] * 2,
        num_blocks_list=[3] * n_layer,
        is_downsample_block=(None, True, True, False, False, False, False, False, False, False, True, False),
        transition_compression_ratio=0.5,
        block_type=["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2,
        **kwargs
    )



# def _init_weights(module, name=None, head_init_scale=1.0):
#     if isinstance(module, nn.Conv2d):
#         nn.init.kaiming_normal_(module.weight)
#     elif isinstance(module, nn.BatchNorm2d):
#         nn.init.constant_(module.weight, 1)
#         nn.init.constant_(module.bias, 0)
#     elif isinstance(module, nn.Linear):
#         nn.init.constant_(module.bias, 0)
#         if name and 'head.' in name:
#             module.weight.data.mul_(head_init_scale)
#             module.bias.data.mul_(head_init_scale)


# def _create_rdnet(variant, pretrained=False, **kwargs):
#     if kwargs.get("pretrained_cfg", "") == "fcmae":
#         # NOTE fcmae pretrained weights have no classifier or final norm-layer (`head.norm`)
#         # This is workaround loading with num_classes=0 w/o removing norm-layer.
#         kwargs.setdefault("pretrained_strict", False)

#     model = build_model_with_cfg(
#         RDNet, variant, pretrained, feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True), **kwargs
#     )
#     return model


# def _cfg(url='', **kwargs):
#     return {
#         "url": url,
#         "num_classes": 1000,
#         "input_size": (3, 224, 224),
#         "crop_pct": 0.9,
#         "interpolation": "bicubic",
#         "mean": IMAGENET_DEFAULT_MEAN,
#         "std": IMAGENET_DEFAULT_STD,
#         "first_conv": "stem.stem.0",
#         "classifier": "head.fc",
#         **kwargs,
#     }


# default_cfgs = generate_default_cfgs({
#     'rdnet_tiny.nv_in1k': _cfg(
#         hf_hub_id='naver-ai/rdnet_tiny.nv_in1k',
#     ),
#     'rdnet_small.nv_in1k': _cfg(
#         hf_hub_id='naver-ai/rdnet_small.nv_in1k',
#     ),
#     'rdnet_base.nv_in1k': _cfg(
#         hf_hub_id='naver-ai/rdnet_base.nv_in1k',
#     ),
#     'rdnet_large.nv_in1k': _cfg(
#         hf_hub_id='naver-ai/rdnet_large.nv_in1k',
#     ),
#     'rdnet_large.nv_in1k_ft_in1k_384': _cfg(
#         hf_hub_id='naver-ai/rdnet_large.nv_in1k_ft_in1k_384',
#         input_size=(3, 384, 384),
#         crop_pct=1.0,
#     ),
# })


# @register_model
# def rdnet_tiny(pretrained=False, **kwargs):
#     n_layer = 7
#     model_args = {
#         "num_init_features": 64,
#         "growth_rates": [64] + [104] + [128] * 4 + [224],
#         "num_blocks_list": [3] * n_layer,
#         "is_downsample_block": (None, True, True, False, False, False, True),
#         "transition_compression_ratio": 0.5,
#         "block_type": ["Block"] + ["Block"] + ["BlockESE"] * 4 + ["BlockESE"],
#     }
#     model = _create_rdnet("rdnet_tiny", pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def rdnet_small(pretrained=False, **kwargs):
#     n_layer = 11
#     model_args = {
#         "num_init_features": 72,
#         "growth_rates": [64] + [128] + [128] * (n_layer - 4) + [240] * 2,
#         "num_blocks_list": [3] * n_layer,
#         "is_downsample_block": (None, True, True, False, False, False, False, False, False, True, False),
#         "transition_compression_ratio": 0.5,
#         "block_type": ["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2,
#     }
#     model = _create_rdnet("rdnet_small", pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def rdnet_base(pretrained=False, **kwargs):
#     n_layer = 11
#     model_args = {
#         "num_init_features": 120,
#         "growth_rates": [96] + [128] + [168] * (n_layer - 4) + [336] * 2,
#         "num_blocks_list": [3] * n_layer,
#         "is_downsample_block": (None, True, True, False, False, False, False, False, False, True, False),
#         "transition_compression_ratio": 0.5,
#         "block_type": ["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2,
#     }
#     model = _create_rdnet("rdnet_base", pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def rdnet_large(pretrained=False, **kwargs):
#     n_layer = 12
#     model_args = {
#         "num_init_features": 144,
#         "growth_rates": [128] + [192] + [256] * (n_layer - 4) + [360] * 2,
#         "num_blocks_list": [3] * n_layer,
#         "is_downsample_block": (None, True, True, False, False, False, False, False, False, False, True, False),
#         "transition_compression_ratio": 0.5,
#         "block_type": ["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2,
#     }
#     model = _create_rdnet("rdnet_large", pretrained=pretrained, **dict(model_args, **kwargs))
#     return model
