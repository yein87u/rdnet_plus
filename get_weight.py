import timm
import torch

model_name = 'rdnet_large.nv_in1k'

# timm.create_model 會自動下載並載入權重
model_timm = timm.create_model(model_name, pretrained=True)

# 獲取 state_dict
pretrained_state_dict = model_timm.state_dict()

# (可選) 將權重保存到本地檔案，供您的 _GetModel 使用
output_path = f"./weight/{model_name}.pth"
torch.save(pretrained_state_dict, output_path)

print(f"預訓練權重已保存至: {output_path}")