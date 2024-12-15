import cv2
import numpy as np
import os
import random

class SaladDataset:
    def __init__(self, path1, path2, path3, frame_step=9):
        """
        Dataset for videos, annotations, and time files.

        Args:
            path1 (str): Path to videos.
            path2 (str): Path to annotations.
            path3 (str): Path to time files.
            frame_step (int): Interval for frame sampling.
        """
        self.max_len = 6000
        self.frame_step = frame_step
        self.file_paths = []
        for folder_path in [path1, path2, path3]:
            for root, dirs, files in os.walk(folder_path):
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    self.file_paths.append(file_path)

    def generate_arrays(self, idl):
        while True:
            idl = sorted(idl, key=lambda x: random.random())
            print(idl)
            for i in idl:
                f, l = self.__getvideo__(i)
                f, l = self.padding(f, l)
                yield (np.array(f), np.array(l))

    def __getvideo__(self, idx):
        """
        Load video frames and annotations.

        Args:
            idx (int): Video index.

        Returns:
            tuple: Frames and labels for the video.
        """
        v = idx
        time = 90 + idx
        ann = 45 + idx
        cap = cv2.VideoCapture(self.file_paths[v])
        frames = []
        frame_idx = 0
        success, image = cap.read()
        
        while success:
            if frame_idx % self.frame_step == 0:  # Pick every `frame_step` frame
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                height, width = image.shape[:2]
                new_width = int(width * 0.65)
                new_height = int(height * 0.65)
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                img = resized_image[((new_height-170)//2):((new_height+180)//2), ((new_width-220)//2):((new_width+250)//2)]
                frames.append(img / 255.0)
            frame_idx += 1
            success, image = cap.read()
        cap.release()

        times = []
        with open(self.file_paths[time], "r") as file:
            for line in file:
                times.append(int(line.split(' ')[0]))

        acts = []
        act = []
#         diction = {
#     # Add Ingredients
#     'add_salt_core': 0, 'add_salt_post': 0, 'add_salt_prep':0,
#     'add_pepper_core': 0, 'add_pepper_prep': 0, 'add_pepper_post': 0,
#     'add_vinegar_prep': 0, 'add_vinegar_core': 0, 'add_vinegar_post': 0,
#     'add_oil_prep': 0, 'add_oil_core': 0, 'add_oil_post': 0,
#     'add_dressing_prep': 0, 'add_dressing_core': 0, 'add_dressing_post': 0,

#     # Mixing Ingredients
#     'mix_dressing_prep': 1, 'mix_dressing_core': 1, 'mix_dressing_post': 1,
#     'mix_ingredients_prep': 1, 'mix_ingredients_core': 1, 'mix_ingredients_post': 1,

#     # Cutting Ingredients
#     'cut_cheese_prep': 2, 'cut_cheese_core': 2, 'cut_cheese_post': 2,
#     'cut_tomato_prep': 2, 'cut_tomato_core': 2, 'cut_tomato_post': 2,
#     'peel_cucumber_prep': 2, 'peel_cucumber_core': 2, 'peel_cucumber_post': 2,
#     'cut_cucumber_prep': 2, 'cut_cucumber_core': 2, 'cut_cucumber_post': 2,
#     'cut_lettuce_prep': 2, 'cut_lettuce_core': 2, 'cut_lettuce_post': 2,

#     # Placing Ingredients in Bowl
#     'place_cucumber_into_bowl_prep': 3, 'place_cucumber_into_bowl_core': 3, 'place_cucumber_into_bowl_post': 3,
#     'place_tomato_into_bowl_prep': 3, 'place_tomato_into_bowl_core': 3, 'place_tomato_into_bowl_post': 3,
#     'place_lettuce_into_bowl_prep': 3, 'place_lettuce_into_bowl_core': 3, 'place_lettuce_into_bowl_post': 3,
#     'place_cheese_into_bowl_prep': 3, 'place_cheese_into_bowl_core': 3, 'place_cheese_into_bowl_post': 3,

#     # Serving
#     'serve_salad': 4,
#     'serve_salad_onto_plate_core': 4, 'serve_salad_onto_plate_prep': 4, 'serve_salad_onto_plate_post': 4
# }
        diction = {'add_salt_prep':0, 
                   'add_salt_core': 1, 
                   'add_salt_post': 2, 
                   'add_pepper_core': 3, 
                   'add_pepper_prep' : 4, 
                   'add_pepper_post':5, 
                   'add_vinegar_prep':6, 
                   'add_vinegar_core':7, 
                   'add_oil_prep':8, 
                   'add_oil_core':9, 
                   'add_oil_post':10, 
                   'mix_dressing_prep' :11, 
                   'mix_dressing_core':12, 
                   'mix_dressing_post':13, 
                   'cut_cheese_core':14, 
                   'cut_cheese_post':15, 
                   'cut_tomato_prep':16, 
                   'cut_tomato_post':17, 
                   'peel_cucumber_core':18, 
                   'peel_cucumber_post':19, 
                   'peel_cucumber_prep':20, 
                   'cut_cucumber_prep':21, 
                   'cut_cucumber_core':22, 
                   'cut_cucumber_post':23, 
                   'cut_lettuce_prep':24, 
                   'cut_lettuce_core':25, 
                   'add_dressing_prep':26, 
                   'add_dressing_core':27, 
                   'add_dressing_post':28, 
                   'mix_ingredients_core':29, 
                   'cut_and_mix_ingredients':30, 
                   'mix_ingredients_prep':31, 
                   'serve_salad':32, 
                   'serve_salad_onto_plate_core':33, 
                   'serve_salad_onto_plate_prep':34, 
                   'serve_salad_onto_plate_post':35, 
                   'add_vinegar_post':36, 
                   'cut_cheese_prep':37, 
                   'cut_tomato_core':38, 
                   'peel_cucumber_prep':39, 
                   'place_cucumber_into_bowl_prep':40, 
                   'place_cucumber_into_bowl_core':41, 
                   'place_cucumber_into_bowl_post':42, 
                   'place_tomato_into_bowl_prep':43, 
                   'place_tomato_into_bowl_core':44, 
                   'place_cucumber_into_bowl_post':45, 
                   'place_cucumber_into_bowl_core':46, 
                   'place_lettuce_into_bowl_core':47, 
                   'place_lettuce_into_bowl_prep':48,
                   'place_lettuce_into_bowl_post':49,
                   'place_cheese_into_bowl_prep':50,
                   'place_cheese_into_bowl_core':51,
                   'place_cheese_into_bowl_post':52,
                   'place_tomato_into_bowl_post':53,
                   'cut_lettuce_post':54,
                   'mix_ingredients_post':55}

        with open(self.file_paths[ann], "r") as file:
            count = 0
            for line in file:
                f = [int(line.split(' ')[0]), int(line.split(' ')[1])]
                type = line.strip().split(' ')[2]
                if type not in ['cut_and_mix_ingredients', 'prepare_dressing', 'serve_salad']:
                    acts.append(f)
                    act.append(diction[type])


        f = [0] * len(frames)
        labels = []
        for t in range(len(times)):
            sampled_idx = t // self.frame_step  # Align time with sampled frames
            if sampled_idx >= len(frames):
                break
            for i in range(len(acts)):
                if acts[i][0] <= times[t] <= acts[i][1]:
                    if f[sampled_idx] == 0:
                        labels.append(act[i])
                    f[sampled_idx] = 1

        aframes = [frames[j] for j, active in enumerate(f) if active == 1]
        return np.array(aframes), np.array(labels)

    def padding(self, frames, labels):
        """
        Pad or truncate frames and labels to max_len.

        Args:
            frames (numpy.ndarray): Frames from the video.
            labels (numpy.ndarray): Labels corresponding to frames.

        Returns:
            tuple: Padded or truncated frames and labels.
        """
        length = len(frames)
        if length > self.max_len:
            pad_frames = frames[:self.max_len]
            pad_labels = labels[:self.max_len]
        else:
            pad_frames = frames
            pad_labels = labels
        return pad_frames, pad_labels
