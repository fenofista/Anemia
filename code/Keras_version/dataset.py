import pandas as pd
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import numpy as np
def get_dataset():
    class AnemiaDataset():
        def __init__(self, annotations_file, img_dir=None, transform=None, target_transform=None):
            self.df = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            image_path = self.df["image_path"][idx]
#             mask_path = self.df["mask_path"][idx]

            image = cv2.imread(image_path)
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
#             for i in range(256):
#                 for j in range(256):
#                     if(gray[i][j]>0):
#                         gray[i][j] = np.log(gray[i][j])
            gray = gray.reshape(1,256,256)
#             mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
            image = image[:,:,::-1]

#             image = np.moveaxis(image, (0,1,2), (1,2,0))

            return gray.copy(), image.copy()
    return AnemiaDataset("data.csv")