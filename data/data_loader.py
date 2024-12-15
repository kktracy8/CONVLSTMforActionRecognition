import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data.motiondata import MotionDataset

image_size_standard = (224,224)
batchsize = 64

motiondata_transform = transforms.Compose([
    transforms.Resize(image_size_standard), 
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.5], std=[0.5]), 
])

motion_dataset = MotionDataset(root_dir='data/datasets/humanmotion_dataset_frames', transform=motiondata_transform)
data_loader = DataLoader(motion_dataset, batch_size=batchsize, shuffle=True)

for images, labels in data_loader:
    print(images.shape) 
    print(labels)        
