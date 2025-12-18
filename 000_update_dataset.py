

import config
import os
import pickle
from collections import Counter

def _GetAllData(dataset_root, save_path):
    data_file = {
        'train':{'path': [], 'label':[]}, 
        'val': {'path': [], 'label': []},
        'test' :{'path': [], 'label':[]}
    }
    
    # 定義要讀取的檔案列表
    modes = ['train', 'test', 'val']
    
    for mode in modes:
        # 檢查檔案是否存在
        file_path = mode+'.txt'
        if not os.path.exists(file_path):
            print(f"警告：檔案 '{file_path}' 不存在，跳過。")
            continue

        try:
            # 以讀取模式 ('r') 開啟檔案
            with open(file_path, 'r', encoding='utf-8') as f:
                # 逐行讀取檔案內容
                for line in f:
                    # 清理行首尾的空白字符（包括換行符）
                    line = line.strip()
                    line = line.replace('PDD271/Sample/', '')
                    path = os.path.join(dataset_root, line)
                    if not line:
                        continue # 跳過空行
                    
                    label = int(line.split('/')[0])-100

                    data_file[mode]['path'].append(path)
                    data_file[mode]['label'].append(label)
        except Exception as e:
            print(f"讀取檔案 {file_path} 時發生錯誤: {e}")
    
    # pickle.dump(data_file, open(save_path, 'wb'))

    # --- 新增統計功能 ---
    print("\n" + "="*30)
    print("📊 數據集類別統計結果:")
    print("="*30)
    
    for mode in modes:
        labels = data_file[mode]['label']
        if not labels:
            print(f"[{mode.upper()}] 無數據")
            continue
            
        # 使用 Counter 統計
        counts = Counter(labels)
        # 依照類別編號排序 (由小到大)
        sorted_counts = dict(sorted(counts.items()))
        
        total = sum(counts.values())
        print(f"--- {mode.upper()} (總數: {total}) ---")
        for label, count in sorted_counts.items():
            print(f"  類別 {label}: {count} 張")
    print("="*30 + "\n")

    return data_file


if __name__ == '__main__':
    args = config.GetArgument()
    data_file = _GetAllData(args.dataset_root, args.pkl_path)

    print(data_file['train']['path'][0:5], data_file['train']['label'][0:5])
    print(data_file['test']['path'][0:5], data_file['test']['label'][0:5])
    print(data_file['val']['path'][0:5], data_file['val']['label'][0:5])



