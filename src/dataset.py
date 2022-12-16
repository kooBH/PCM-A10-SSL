import os, glob
import torch
import librosa as rs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = hp.data.root
        self.GT = {
            "degree0":0,
            "degree20":1,
            "degree40":2,
            "degree60":3,
            "degree80":4,
            "degree100":5,
            "degree120":6,
            "degree140":7,
            "degree160":8,
            "degree180":9
        }
        self.list_data = glob.glob(os.path.join(root,"**","*.wav"))

    def __getitem__(self, index):
        path_item = self.list_data[index]
        dir_item = path_item.split("/")[-2]

        label = self.GT[dir_item]

        raw = rs.load(path_item,sr=16000,mono=False)[0]
        data = rs.stft(raw,n_fft = 512,hop_length=128)

        return torch.from_numpy(data)

    def __len__(self):
        return len(self.list_data)