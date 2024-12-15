import cv2
import numpy as np
import os
import random
import time
import videos
import ann
#import tensorflow as tf

class SaladDataset:
    def __init__(self, path1, path2, path3):
        self.max_length = 9500
        self.file_paths = []
        for folder_path in [path1, path2, path3]:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.file_paths.append(file_path)

    def generate_arrays(self, idl):
        while True:
            idl = sorted(idl, key=lambda x: random.random())
            for i in idl:
                f, l = self.__getvideo__(i)
                print(f.shape)
                print(len(l))
                f, l = self.padding(f, l)
                print(f.shape)
                print(len(l))
                yield (np.array(f), l) #returns as a tuple, but we can update this to a tensor later
    
    def __getvideo__(self, idx):
        v = idx
        time = 1 + idx
        ann = 2 + idx
        cap = cv2.VideoCapture(self.file_paths[v])
        frames = []
        success,image = cap.read()
        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            width, height = image.shape
            img = image[((height-350)//2):((height+50)//2), ((width-70)//2):((width+510)//2)]
            w, h = img.shape
            frames.append(img)  
            success,image = cap.read()
        times = []
        with open(self.file_paths[time], "r") as file:
            for line in file:
                times.append(int(line.split(' ')[0]))
        acts = []
        act = []
        with open(self.file_paths[ann], "r") as file:
            count = 0
            for line in file:
                if count >= 2:
                    acts.append([int(line.split(' ')[0]), int(line.split(' ')[1])])
                    act.append(line.strip().split(' ')[2])
                count += 1
        f = []
        count = 0
        labels = []
        for t in range(len(times)):
            f.append(0)
            for i in range(len(acts)):
                if acts[i][0] <= times[t] <= acts[i][1]:
                    if f[t] == 0:
                        count += 1
                        labels.append(act[i])
                    f[t] = 1
        aframes = []
        for j in range(len(times)):
            if f[j] == 1:
                aframes.append(frames[j])
        fram = np.array(aframes)
        #dataset = tf.data.Dataset.from_tensor_slices((aframes, labels)) if we want to use tensorflow
        return fram, labels
        #returns of shape(9176, 200, 290)
    
    #padding might need to be tweaked a bit
    def padding(self, frames, labels):
        length = len(frames)
        #if length < self.max_len:
        #    size = self.max_len - length
        #    pad_frames = np.pad(frames, ((0, padding_size), (0, 0), (0, 0), (0, 0)), 'constant')
        #    pad_labels = np.pad(labels, (0, padding_size), 'constant', constant_values=-1)
        if length > self.max_len:
            pad_frames = frames[:self.max_len]
            pad_labels = labels[:self.max_len]
        else:
            pad_frames = frames
            pad_labels = labels
        return pad_frames, pad_labels