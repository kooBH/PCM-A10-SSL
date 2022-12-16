import torch
import torch.nn as nn

 def LogMelGccExtractor(self, sig):
        def logmel(sig):
            #pdb.set_trace()
            S = np.abs(librosa.stft(y=sig,
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2        
        
            # S_mel = np.dot(self.melW, S).T
            S = librosa.power_to_db(S**2, ref=1.0, amin=1e-10, top_db=None)
            S = np.expand_dims(S, axis=0)

            return S

        def gcc_phat(sig, refsig):

            #pdb.set_trace()
            Px = librosa.stft(y=sig,
                            n_fft=self.nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window, 
                            pad_mode='reflect')

            Px_ref = librosa.stft(y=refsig,
                                n_fft=self.nfft,
                                hop_length=self.hopsize,
                                center=True,
                                window=self.window,
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
            
            #pdb.set_trace()
            feature_logmel = np.concatenate(feature_logmel, axis=0)
            feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
            feature = np.concatenate([feature_logmel, np.expand_dims(feature_gcc_phat, axis=0)])

            return feature
        
        return transform(sig)

class CRNN9(nn.Module):
    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), pretrained_path=None):
        
        super().__init__()
        self.class_num = class_num
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.interp_ratio = 8
        
        self.conv_block1 = ConvBlock(in_channels=3, out_channels=128)    # 1: 7, 128     2: 7, 64
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=256)  # 1: 128, 256   2: 64, 256
        self.conv_block3 = ConvBlock(in_channels=256, out_channels=512)

        #self.gru = nn.GRU(input_size=512, hidden_size=256, 
        #    num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        self.azimuth_fc = nn.Linear(512, class_num, bias=True)

        self.init_weights()

    def init_weights(self):

        #init_gru(self.gru)
        init_layer(self.azimuth_fc)


    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        
        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''
        # 
        # self.gru.flatten_parameters()
        
        '''if pack padded'''
        # '''else'''
        #(x, _) = self.gru(x)

        azimuth_output = self.azimuth_fc(x)
        # Interpolate
        output = interpolate(azimuth_output, self.interp_ratio) 

        return output

class pretrained_CRNN8(CRNN9):

    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), pretrained_path=None):
        super().__init__(class_num, pool_type, pool_size, pretrained_path=pretrained_path)
        if pretrained_path:
            self.load_weights(pretrained_path)

        self.gru = nn.GRU(input_size=512, hidden_size=256, 
            num_layers=1, batch_first=True, bidirectional=True)

        init_gru(self.gru)
        init_layer(self.azimuth_fc)

    def load_weights(self, pretrained_path):

        model = CRNN9(self.class_num, self.pool_type, self.pool_size)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3
