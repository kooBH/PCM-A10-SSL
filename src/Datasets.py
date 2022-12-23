import os, glob
import torch
import librosa as rs
import numpy as np

def Audio_Collate(batch):
   
    data, class_num = list(zip(*batch))
    data_len = torch.LongTensor(np.array([x.size(1) for x in data if x.size(1)!=1]))

    max_len = max(data_len)
    wrong_indices = []
    
    B = len(data)
    inputs = torch.zeros(B, data[0].shape[0], max_len, 257)
    labels = torch.zeros(B, 10)
    j = 0
    '''zero pad'''    
    for i in range(B):
        inputs[j, : , :data[i].size(1),:] = data[i]
        labels[j, class_num[i]] = 1.0
        j += 1

    data = (inputs, labels)
    return data

class DatasetSSL(torch.utils.data.Dataset):
    def __init__(self, hp, is_train):
        self.n_fft = hp.audio.n_fft
        self.n_hop = hp.audio.n_hop
        self.sr = hp.audio.sr
        self.dB = hp.feature.dB
        self.cc = hp.feature.cc
        self.phat = hp.feature.phat

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
        if is_train : 
            self.list_data = glob.glob(os.path.join(hp.data.train,"*","*.wav"),recursive=True)
        else :
            self.list_data = glob.glob(os.path.join(hp.data.test,"*","*.wav"),recursive=True)

    def feature2021(self, sig):
        def logmel(sig):

            #pdb.set_trace()
            S = np.abs(rs.stft(y=sig,
                                    n_fft=self.n_fft,
                                    hop_length=self.n_hop,
                                    center=True,
                                    pad_mode='reflect'))**2        
        
            # S_mel = np.dot(self.melW, S).T
            S = rs.power_to_db(S**2, ref=1.0, amin=1e-10, top_db=None)
            S = np.expand_dims(S, axis=0)

            return S

        def gcc_phat(sig, refsig):

            Px = rs.stft(y=sig,
                            n_fft=self.n_fft,
                            hop_length=self.n_hop,
                            center=True,
                            pad_mode='reflect')

            Px_ref = rs.stft(y=refsig,
                                n_fft=self.n_fft,
                                hop_length=self.n_hop,
                                center=True,
                                pad_mode='reflect')
        
            R = Px*np.conj(Px_ref)
            return R

        def transform(audio):
            channel_num = audio.shape[0]
            feature_logmel = []
            feature_gcc_phat = []
            for n in range(channel_num):
                feature_logmel.append(logmel(audio[n]))
                for m in range(n+1, channel_num):
                    feature_gcc_phat.append(
                        gcc_phat(sig=audio[m], refsig=audio[n]))
            
            feature_logmel = np.concatenate(feature_logmel, axis=0)
            feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
            feature = np.concatenate([feature_logmel, np.expand_dims(feature_gcc_phat, axis=0)])

            return feature
        
        return transform(sig)

    def feature2022(self, sig):
        def logdb(sig):

            #pdb.set_trace()
            S = np.abs(rs.stft(y=sig,
                                    n_fft=self.n_fft,
                                    hop_length=self.n_hop,
                                    center=True,
                                    pad_mode='reflect'))**2        
        
            # S_mel = np.dot(self.melW, S).T
            S = rs.power_to_db(S**2, ref=1.0, amin=1e-10, top_db=None)
            S = np.expand_dims(S, axis=0)

            return S

        def gcc_phat(sig, refsig):
            Px = rs.stft(y=sig,
                            n_fft=self.n_fft,
                            hop_length=self.n_hop,
                            center=True,
                            pad_mode='reflect')

            Px_ref = rs.stft(y=refsig,
                                n_fft=self.n_fft,
                                hop_length=self.n_hop,
                                center=True,
                                pad_mode='reflect')
        
            R = Px*np.conj(Px_ref)
            if self.phat : 
                R = R/(np.abs(R)+1e-13)
            R = np.stack([R.real,R.imag])
            return R

        def transform(audio):
            channel_num = audio.shape[0]
            feature = None

            if self.dB : 
                feature_logdb = []
                for n in range(channel_num):
                    feature_logdb.append(logdb(audio[n]))
                feature_logdb = np.concatenate(feature_logdb, axis=0)

                if feature is None : 
                    feature = feature_logdb
                else : 
                    feature = np.concatenate([feature, feature_logdb])


            if self.cc : 
                feature_gcc_phat = []
                for n in range(channel_num):
                    for m in range(n+1, channel_num):
                        feature_gcc_phat.append(
                            gcc_phat(sig=audio[m], refsig=audio[n]))
                feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
                if feature is None : 
                    feature = feature_gcc_phat
                else : 
                    feature = np.concatenate([feature, feature_gcc_phat])

            return feature
        
        return transform(sig)

    def __getitem__(self, index):
        path_item = self.list_data[index]
        dir_item = path_item.split("/")[-2]
        label = self.GT[dir_item]

        raw = rs.load(path_item,sr=self.sr,mono=False,dtype=np.float32)[0]
        # trim
        if raw.shape[1] >80000:
            raw= audio[:,:80000]

        # [C, T, F]
        feature = self.feature2022(raw)
        return torch.FloatTensor(feature).transpose(1,2), label

    def __len__(self):
        return len(self.list_data)