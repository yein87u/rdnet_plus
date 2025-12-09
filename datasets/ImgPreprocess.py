import torch
import pickle
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, args, phase='train'):
        self.args = args
        self.phase = phase
        self.image_size = self.args.image_size

        self._read_path_label()
        self._get_mean_std()
        self._get_label_list()
        self._get_num_per_cls()
        self._setup_transforms(self.phase)
    
    def _read_path_label(self):
        pkl = pickle.load(open(self.args.pkl_path, 'rb'))
        self.data = pkl[self.phase]
        self.dataset_size = len(self.data['path'])
    
    def _get_num_per_cls(self):
        self.cls_num_list = np.zeros(self.args.classes)
        for sample in self.data['label']:
            self.cls_num_list[sample] += 1
    
    def _get_label_list(self):
        self.label_list = list(self.data['label'])

    def _get_mean_std(self):
        if self.phase == 'train':
            self.mean = np.zeros(3, dtype=np.float32)   
            self.std = np.zeros(3, dtype=np.float32)
            numSamples = 0
            for img_path in tqdm(self.data['path'], desc=f"計算 {self.phase} mean/std", ncols=100, leave=False):
                image = cv2.imread(img_path)
                if image is None:
                    print(f"⚠️ 無法讀取圖片：{img_path}")
                    continue
                image = image / 255.0

                self.mean[0] += np.mean(image[...,0])
                self.mean[1] += np.mean(image[...,1])
                self.mean[2] += np.mean(image[...,2])
                self.std[0] += np.std(image[...,0])
                self.std[1] += np.std(image[...,1])
                self.std[2] += np.std(image[...,2])
                
                numSamples += 1
            self.mean /= numSamples
            self.std /= numSamples
            self.args.mean = self.mean
            self.args.std = self.std
        elif self.phase == 'val' or self.phase == 'test':
            self.mean = self.args.mean
            self.std = self.args.std
        self.mean = list(self.mean)
        self.std = list(self.std)

        # self.mean = [np.float32(0.36649305), np.float32(0.51296484), np.float32(0.41327184)]
        # self.std = [np.float32(0.18601118), np.float32(0.17255422), np.float32(0.17981789)]
        # self.args.mean = self.mean
        # self.args.std = self.std
        # print(f'mean: {self.mean}\nstd: {self.std}')
    
    def _setup_transforms(self, phase): 
        self.phase = phase
        if self.phase == 'train': #to do CLAHE
            self.transforms = A.Compose([
                A.Resize(
                        self.args.image_size[0],
                        self.args.image_size[1],), 

                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),

                # # 取代 CLAHE：亮度/對比增強（模擬不同光線） 
                # A.RandomBrightnessContrast(
                #     brightness_limit=0.35, 
                #     contrast_limit=0.35, 
                #     p=0.8), 
                
                # # 顏色增強：模擬不同相機/季節的葉色變化 
                # A.HueSaturationValue(
                #     hue_shift_limit=12, 
                #     sat_shift_limit=20, 
                #     val_shift_limit=15, 
                #     p=0.6),
                
                A.HorizontalFlip(p=0.5), 
                A.VerticalFlip(p=0.2), 
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, p=0.5),
                
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])

        elif self.phase == 'test' or self.phase == 'val':
            self.transforms = A.Compose([
                A.Resize(
                        self.args.image_size[0],
                        self.args.image_size[1],), 
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
    
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        img_path = self.data['path'][idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']

        label = self.data['label'][idx]
        label = torch.tensor(label, dtype=torch.int64)
        return image, label
    


class data_prefetcher():
    def __init__(self, args, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(args.device)
        self.preload()


    def preload(self):
        try:
            self.next_image, self.next_label = next(self.loader)
        except StopIteration:
            self.next_image = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image = self.next_image
        label = self.next_label 
        self.preload()
        return image, label