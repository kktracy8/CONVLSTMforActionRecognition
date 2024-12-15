import os
from PIL import Image
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
    
        for img_file in os.listdir(root_dir):
            img_path = os.path.join(root_dir, img_file)
            self.image_paths.append(img_path)

            class_label = img_file.split('_')[0]  
            self.labels.append(class_label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L") 

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label 