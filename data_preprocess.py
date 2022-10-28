import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, data_path, labels, N_frame, IMG_SIZE, transform):
        self.transform = transform
        self.ids = data_path
        self.labels = labels
        self.n_frame = N_frame
        self.img_size = IMG_SIZE

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.ids[idx])
        label = self.labels[idx]
        v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_list = np.linspace(0, v_len - 1, self.n_frame, dtype=np.int16)
        frames = []
        # repeat the short video to n_frame
        if v_len < self.n_frame :
            store = []
            for fn in range(v_len):
                ret, frame = cap.read()
                store.append(frame)
            temp = self.n_frame // v_len
            store = store * temp
            if len(store) < self.n_frame:
                for i in store:
                    store.append(i)
                    if len(store) == self.n_frame:
                        break
            for frame in store:
                frames.append(frame)
        
        else:
            for fn in range(v_len):
                ret, frame = cap.read()
                if not ret:
                    break
                if (fn in frame_list):
                    frames.append(frame)
        cap.release()
        
        frames_tr = []
        for frame in frames:
            frame = self.transform(frame)
            frames_tr.append(frame)
        
        frames_tr = torch.stack(frames_tr)
        
        if label != None:
            return frames_tr, label 
        if label == None:
            return frames_tr, self.ids[idx]






