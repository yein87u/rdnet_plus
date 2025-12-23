# rdnet_plus
## 影像視訊處理專案(植物疾病檢測)
本研究以 PDD271 大規模植物疾病資料集為基礎，提出一套基於 RDNet 預訓練模型的改良式植物疾病分類架構，旨在提升模型對病斑區域的關注能力與整體辨識準確率。文件: https://drive.google.com/file/d/1N_Qva-kB2ZXHc5sh8MFndo4BbTWljfZS/view?usp=drive_link

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

### 環境建置
執行
```bash
conda env create -f environment.yml
```

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

### 參考文獻
[1]	X. Liu, W. Min, S. Mei, L. Wang, and S. Jiang, “Plant disease recognition: a large-Scale benchmark dataset and a visual region and loss reweighting approach,” IEEE Transactions on Image Processing, vol. 30, pp. 2003-2015, 2021.
[2]	E. Moupojou, F. Retraint, H. Tapamo, M. Nkenlifack, C. Kacfah and A. Tagne, “Segment anything model and fully convolutional data description for plant multi-disease detection on field images,” IEEE Access, vol. 12, pp. 102592-102605, 2024.
[3]	A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár and R. Girshick, “Segment anything”, arXiv preprint arXiv:2304.02643, 2023.
[4]	P. Liznerski, L. Ruff, R. A. Vandermeulen, B. Joe Franks, M. Kloft and K.-R. Müller, “Explainable deep one-class classification,” arXiv preprint arXiv:2007.01760, 2020.
[5]	Z. Xiao, Y. Shi, G. Zhu, J. Xiong, and J. Wu, “Leaf disease detection based on lightweight deep residual network and attention mechanism,” IEEE Access, vol. 11, pp. 48248–48258, 2023.
[6]	N. Ma, X. Zhang, H.-T. Zheng, and J. Sun, “ShuffleNet v2: practical guidelines for efficient CNN architecture design,” in Proc. Eur. Conf. Comput. Vis. Cham, Switzerland: Springer, pp. 116–131, 2018.
[7]	S. Ahmed, M. B. Hasan, T. Ahmed, M. R. K. Sony, and M. H. Kabir, “Less is more: lighter and faster deep neural architecture for tomato leaf disease classification,” IEEE Access, vol. 10, pp. 68868–68884, 2022.
[8]	M. Hassam, M. A. Khan, A. Armghan, S. A. Althubiti, M. Alhaisoni, A. Alqahtani, S. Kadry, and Y. Kim, “A single stream modified MobileNet V2 and whale controlled entropy based optimization framework for citrus fruit diseases recognition,” IEEE Access, vol. 10, pp. 91828–91839, 2022.
[9]	D. Kim, B. Heo, and D. Han, “DenseNets reloaded: paradigm shift beyond resnets and vits,” arXiv preprint arXiv:2403.19588, 2024.
[10]	K. He, X. Zhang, S. Ren and J. Sun “Deep residual learning for image recognition,” IEEE Conference on Computer Vision and Pattern Recognition (CVPR), doi: 10.1109/CVPR.2016.90, 2016.
[11]	K. Cao, C. Wei, A. Gaidon, N. Arechiga and T. Ma, “Learning imbalanced datasets with label-distribution-aware margin loss,” arXiv preprint arXiv:1906.07413, 2019.

