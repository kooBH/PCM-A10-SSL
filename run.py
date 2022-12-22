import torch
import os,sys,glob
sys.path.append("./src")
import argparse

from models import CRNNv2
import numpy as np
import librosa as rs

#from common import run, get_model

def preprocess(sig):
        def logdb(sig):
            #pdb.set_trace()
            S = np.abs(rs.stft(y=sig,
                                    n_fft=512,
                                    hop_length=128,
                                    center=True,
                                    pad_mode='reflect'))**2        
        
            # S_mel = np.dot(self.melW, S).T
            S = rs.power_to_db(S**2, ref=1.0, amin=1e-10, top_db=None)
            S = np.expand_dims(S, axis=0)

            return S

        def gcc_phat(sig, refsig):
            Px = rs.stft(y=sig,
                            n_fft=512,
                            hop_length=128,
                            center=True,
                            pad_mode='reflect')

            Px_ref = rs.stft(y=refsig,
                                n_fft=512,
                                hop_length=128,
                                center=True,
                                pad_mode='reflect')
        
            R = Px*np.conj(Px_ref)
            R = R/(np.abs(R)+1e-13)
            R = np.stack([R.real,R.imag])
            return R

        def transform(audio):
            channel_num = audio.shape[0]
            feature_logdb = []

            for n in range(channel_num):
                feature_logdb.append(logdb(audio[n]))
            feature_logdb = np.concatenate(feature_logdb, axis=0)

            feature_gcc_phat = []
            for n in range(channel_num):
                for m in range(n+1, channel_num):
                    feature_gcc_phat.append(
                        gcc_phat(sig=audio[m], refsig=audio[n]))
            feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
            feature = np.concatenate([feature_logdb, feature_gcc_phat])

            return feature
        
        return transform(sig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('--dir_in','-i',type=str,required=True)
    args = parser.parse_args()

    dir_in = args.dir_in
    device = args.device
    torch.cuda.set_device(device)

    ## Model
    model = CRNNv2(4,
        pool_type = "avg",
        last_activation = "Sigmoid" 
        ).to(device)

    model.load_state_dict(torch.load("chkpt/bestmodel.pt", map_location=device))


    list_target = glob.glob(os.path.join(dir_in,"**","*.wav"),recursive=True)

    print("n_data : {}".format(len(list_target)))

    model.eval()
    with torch.no_grad():
        for path in list_target : 
            x = rs.load(path,mono=False,sr=16000)[0]
            feature = preprocess(x)
            feature = np.expand_dims(feature,0)
            feature = torch.from_numpy(feature)
            feature = feature.to(device)

            output = model(feature)

            estim = output.argmax()

            print("{}'s estimated direction is {}".format(path,20*(estim)))
