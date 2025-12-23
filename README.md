# rdnet_plus
## 影像視訊處理專案(植物疾病檢測)
本研究以 PDD271 大規模植物疾病資料集為基礎，提出一套基於 RDNet 預訓練模型的改良式植物疾病分類架構，旨在提升模型對病斑區域的關注能力與整體辨識準確率。

### 模型設計
研究於 RDNet 主幹中引入 dropout 以提升泛化能力，並在 ESE 模組後加入空間注意力機制(Spatial Attention, SA)，強化模型對關鍵病灶區域的空間聚焦效果；同時，將分類頭由傳統 Linear 層替換為 NormLinear，以改善特徵分佈穩定性並增強類別間的判別能力。

### 訓練策略
本研究採用三階段訓練流程。第一階段凍結主幹權重，僅針對分類頭與空間注意力模組以較高學習率進行訓練；第二階段解凍主幹網絡，透過較小學習率進行全模型微調；第三階段則引入 LDAM Loss，針對資料集中尾部類別進行重加權學習，以緩解長尾分佈對模型效能的影響。

### 實驗結果
實驗結果顯示，所提出的方法在PDD271資料集上的分類準確率由原先的90.68%提升至93.29%，在不顯著增加模型複雜度的前提下，有效提升了對少樣本類別與實際農田場景的辨識能力。

### 復現本研究
請先下載checkpoints

checkpoints download: https://drive.google.com/drive/folders/1gls3_weVCN82npb5JQcLuQPnlrHBegnj?usp=drive_link

於根目錄下創建/new_checkpoint，將下載好的checkpoints放進資料夾中

rdnet_base.nv_in1k_bz16_timm: 直接使用timm.create_model創建模型, 包含分類頭、FC
rdnet_base_SAttention_bz16_origin_fc: 重新設置預訓練的fc
rdnet_base_SAttention_bz16_noSA: 使用本研究提出的rdnet_plus(rdnet_base_SAttention)模型，但需要註解models/pretrain_rdnet.py中的_inject_spatial_attention()
rdnet_base_SAttention_bz16_general_DA: 直接使用本研究提出的rdnet_plus(rdnet_base_SAttention)模型
rdnet_base_SAttention_bz16_HSV_brightness: 直接使用本研究提出的rdnet_plus(rdnet_base_SAttention)模型
rdnet_base_SAttention_bz16_noDA: 直接使用本研究提出的rdnet_plus(rdnet_base_SAttention)模型
rdnet_base_SAttention_bz16_final: 直接使用本研究提出的rdnet_plus(rdnet_base_SAttention)模型(此為最終版本)

### 實驗結果
本研究與資料集論文與各個預訓練模型進行比較，皆大幅領先其他模型與方法，如下表所示。

|Model|Accuracy(%)|
|-|-|
|resnet50|84.11|
|resnet101|82.59|
|efficientnet_b0|86.67|
|efficientnet_b1|87.21|
|efficientnet_b2|85.29|
|mobilenetv1_100|84.75|
|SeNet154[1]|85.58|
|RDNet[9]|90.68|
|RDNet_plus(our)|93.29|