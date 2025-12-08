import os
import config
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def _GetAllData(dataset_root):
    all_data = {'img_path':[], 'label':[]}
    print(dataset_root)
    count_folder = len(os.listdir(dataset_root))-1
    for i in tqdm(range(100, 100+count_folder), desc="Get Image Path: ", total=count_folder, ncols=100):
        sub_folder = os.path.join(dataset_root, str(i))

        # 如果子資料夾不存在則繼續
        if not os.path.isdir(sub_folder):
            print(f"無類別 {i} 資料夾...")
            continue

        # 若為1，則資料夾內還是一個資料夾，需更新path
        if len(os.listdir(sub_folder)) == 1:
            sub_folder = os.path.join(sub_folder, str(i))
        
        for img in os.listdir(sub_folder):
            img_path = os.path.join(sub_folder, img)
            all_data['img_path'].append(img_path)
            all_data['label'].append(i-100)
    
    # print(all_data['img_path'][60:65])
    # print(all_data['label'][60:65])

    return all_data

    
def create_annotation_file(all_data, save_path):
    data_file = {
        'train':{'path': [], 'label':[]}, 
        'val': {'path': [], 'label': []},
        'test' :{'path': [], 'label':[]}
    }
    # 先切出 80% train + 20% temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_data['img_path'],
        all_data['label'],
        test_size=0.2,            # 80% train + 20% temp
        stratify=all_data['label'],
        random_state=42
    )

    # 再把 20% temp 切成 10% val + 10% test（即 1:1）
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,            # 把 temp 平分
        stratify=temp_labels,
        random_state=42
    )

    # 存入 dict
    data_file['train']['path'] = train_paths
    data_file['train']['label'] = train_labels
    data_file['val']['path'] = val_paths
    data_file['val']['label'] = val_labels
    data_file['test']['path'] = test_paths
    data_file['test']['label'] = test_labels

    pickle.dump(data_file, open(save_path, 'wb'))
    print(len(data_file['train']['path']), len(data_file['train']['label']), data_file['train']['path'][0], data_file['train']['label'][0])
    print(len(data_file['val']['path']), len(data_file['val']['label']), data_file['val']['path'][0], data_file['val']['label'][0])
    print(len(data_file['test']['path']), len(data_file['test']['label']), data_file['test']['path'][0], data_file['test']['label'][0])

    
    from collections import Counter
    import pandas as pd
    # Step 5️⃣: 檢查每個類別的分布
    print("\n📊 類別分布檢查：")
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    # 合併成 DataFrame 方便看
    all_labels = sorted(set(all_data['label']))
    df = pd.DataFrame({
        'Class': all_labels,
        'Train': [train_counts.get(c, 0) for c in all_labels],
        'Val': [val_counts.get(c, 0) for c in all_labels],
        'Test': [test_counts.get(c, 0) for c in all_labels]
    })
    df['Total'] = df['Train'] + df['Val'] + df['Test']
    df['Train%'] = (df['Train'] / df['Total'] * 100).round(1)
    df['Val%'] = (df['Val'] / df['Total'] * 100).round(1)
    df['Test%'] = (df['Test'] / df['Total'] * 100).round(1)

    print(df)

if __name__ == '__main__':
    args = config.GetArgument()
    all_data = _GetAllData(args.dataset_root)
    create_annotation_file(all_data, args.pkl_path)
    