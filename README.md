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
checkpoints download: 
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

<table class="MsoTableGrid" border="0" cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:none;mso-yfti-tbllook:1184;mso-padding-alt:
 0cm 5.4pt 0cm 5.4pt;mso-border-insideh:none;mso-border-insidev:none">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes">
  <td width="189" style="width:141.5pt;border-top:double windowtext 1.5pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><b><span lang="EN-US">Models<o:p></o:p></span></b></p>
  </td>
  <td width="112" style="width:83.65pt;border-top:double windowtext 1.5pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><b><span lang="EN-US">Accuracy (%)<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1">
  <td width="189" style="width:141.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US">resnet50<o:p></o:p></span></p>
  </td>
  <td width="112" style="width:83.65pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US">84.11<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2">
  <td width="189" style="width:141.5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US" style="mso-bidi-font-weight:bold">resnet101<o:p></o:p></span></p>
  </td>
  <td width="112" style="width:83.65pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US">82.59<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3">
  <td width="189" style="width:141.5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US" style="mso-bidi-font-weight:bold">efficientnet_b0<o:p></o:p></span></p>
  </td>
  <td width="112" style="width:83.65pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US">86.67<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4">
  <td width="189" style="width:141.5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US" style="mso-bidi-font-weight:bold">efficientnet_b1<o:p></o:p></span></p>
  </td>
  <td width="112" style="width:83.65pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US">87.21<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5">
  <td width="189" style="width:141.5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US" style="mso-bidi-font-weight:bold">efficientnet_b2<o:p></o:p></span></p>
  </td>
  <td width="112" style="width:83.65pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US">85.29<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6">
  <td width="189" style="width:141.5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US" style="mso-bidi-font-weight:bold">mobilenetv1_100<o:p></o:p></span></p>
  </td>
  <td width="112" style="width:83.65pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US">84.75<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:7">
  <td width="189" style="width:141.5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US" style="mso-bidi-font-weight:bold">SeNet154[1]<o:p></o:p></span></p>
  </td>
  <td width="112" style="width:83.65pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US" style="mso-bidi-font-weight:bold">85.58</span><span lang="EN-US"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:8">
  <td width="189" style="width:141.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><i style='mso-bidi-font-style:normal'><span
   lang=EN-US style='font-family:"Cambria Math",serif'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="bi"/></m:rPr>RDNet</m:r></span></i></b></m:oMath><![endif]--><!--[if !msEquation]--><span lang="EN-US" style="font-size:12.0pt;line-height:115%;font-family:&quot;Aptos&quot;,sans-serif;
  mso-ascii-theme-font:minor-latin;mso-fareast-font-family:新細明體;mso-fareast-theme-font:
  minor-fareast;mso-hansi-theme-font:minor-latin;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-bidi-theme-font:minor-bidi;position:relative;top:6.0pt;mso-text-raise:
  -6.0pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-TW;mso-bidi-language:
  AR-SA"><!--[if gte vml 1]><v:shapetype id="_x0000_t75" coordsize="21600,21600"
   o:spt="75" o:preferrelative="t" path="m@4@5l@4@11@9@11@9@5xe" filled="f"
   stroked="f">
   <v:stroke joinstyle="miter"/>
   <v:formulas>
    <v:f eqn="if lineDrawn pixelLineWidth 0"/>
    <v:f eqn="sum @0 1 0"/>
    <v:f eqn="sum 0 0 @1"/>
    <v:f eqn="prod @2 1 2"/>
    <v:f eqn="prod @3 21600 pixelWidth"/>
    <v:f eqn="prod @3 21600 pixelHeight"/>
    <v:f eqn="sum @0 0 1"/>
    <v:f eqn="prod @6 1 2"/>
    <v:f eqn="prod @7 21600 pixelWidth"/>
    <v:f eqn="sum @8 21600 0"/>
    <v:f eqn="prod @7 21600 pixelHeight"/>
    <v:f eqn="sum @10 21600 0"/>
   </v:formulas>
   <v:path o:extrusionok="f" gradientshapeok="t" o:connecttype="rect"/>
   <o:lock v:ext="edit" aspectratio="t"/>
  </v:shapetype><v:shape id="_x0000_i1025" type="#_x0000_t75" style='width:38.4pt;
   height:21pt'>
   <v:imagedata src="Models.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><!--[if !vml]--><img width="51" height="28" src="Models.files/image002.png" v:shapes="_x0000_i1025"><!--[endif]--></span><!--[endif]--><span lang="EN-US" style="mso-bidi-font-weight:bold">[9]<o:p></o:p></span></p>
  </td>
  <td width="112" style="width:83.65pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><span lang="EN-US">90.68<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:9;mso-yfti-lastrow:yes">
  <td width="189" style="width:141.5pt;border:none;border-bottom:double windowtext 1.5pt;
  mso-border-top-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><i style='mso-bidi-font-style:normal'><span
   lang=EN-US style='font-family:"Cambria Math",serif'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="bi"/></m:rPr>RDNet</m:r><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="bi"/></m:rPr>_</m:r><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="bi"/></m:rPr>plus</m:r></span></i></b></m:oMath><![endif]--><!--[if !msEquation]--><span lang="EN-US" style="font-size:12.0pt;line-height:115%;font-family:&quot;Aptos&quot;,sans-serif;
  mso-ascii-theme-font:minor-latin;mso-fareast-font-family:新細明體;mso-fareast-theme-font:
  minor-fareast;mso-hansi-theme-font:minor-latin;mso-bidi-font-family:&quot;Times New Roman&quot;;
  mso-bidi-theme-font:minor-bidi;position:relative;top:6.0pt;mso-text-raise:
  -6.0pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-TW;mso-bidi-language:
  AR-SA"><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:68.4pt;height:21pt'>
   <v:imagedata src="Models.files/image003.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><!--[if !vml]--><img width="91" height="28" src="Models.files/image004.png" v:shapes="_x0000_i1025"><!--[endif]--></span><!--[endif]--><span lang="EN-US" style="mso-bidi-font-weight:bold">(our)</span><span lang="EN-US"><o:p></o:p></span></p>
  </td>
  <td width="112" style="width:83.65pt;border:none;border-bottom:double windowtext 1.5pt;
  mso-border-top-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="margin-bottom:8.0pt;line-height:115%"><b><span lang="EN-US">93.29<o:p></o:p></span></b></p>
  </td>
 </tr>
</tbody></table>
