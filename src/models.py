import torch
import torch.nn as nn
from modules import *
from asteroid_filterbanks import ParamSincFB

class CRNN(nn.Module):
    def __init__(self, c_in, class_num=10, pool_type='avg', pool_size=(2,2), pretrained_path=None,last_activation="Sigmoid"):
        super().__init__()

        self.class_num = class_num
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.interp_ratio = 7
        
        self.conv_block0 = ConvBlock(in_channels=c_in, out_channels=64)    # 1: 7, 128     2: 7, 64
        self.conv_block1 = ConvBlock(in_channels=64, out_channels=256)  # 1: 128, 256   2: 64, 256
        self.conv_block2 = ConvBlock(in_channels=256, out_channels=512)

        self.gate_block0 = ConvBlock(in_channels=4, out_channels=64)
        self.gate_block1 = ConvBlock(in_channels=64, out_channels=256)
        self.gate_block2 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=256, 
            num_layers=1, dropout=0.3, batch_first=True, bidirectional=True)

        #self.azimuth_fc = nn.Linear(511, class_num, bias=True)
        self.azimuth_fc0 = nn.Linear(512, 128, bias=True)
        self.azimuth_fc1 = nn.Linear(128, class_num, bias=True)

        self.init_weights()

        if last_activation == "Sigmoid" :
            self.last_activation = nn.Sigmoid()
        elif last_activation == "Softmax" : 
            self.last_activation = nn.Softmax()
        else : 
            self.last_activation = nn.Sigmoid()

    def init_weights(self):

        init_gru(self.gru)
        #init_layer(self.azimuth_fc)
        init_layer(self.azimuth_fc0)
        init_layer(self.azimuth_fc1)

    def forward(self, x):
        #pdb.set_trace() 
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        gate = self.gate_block0(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block0(x, self.pool_type, pool_size=self.pool_size)
        x = x * torch.sigmoid(gate)

        gate = self.gate_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = x * torch.sigmoid(gate)

        gate = self.gate_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = x * torch.sigmoid(gate)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=2)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=2)
        '''(batch_size, feature_maps, time_steps)'''
        
        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        self.gru.flatten_parameters()
        (x, _) = self.gru(x)
        
        x = self.azimuth_fc0(x)
        azimuth_output = self.azimuth_fc1(x)
        #azimuth_output = self.azimuth_fc(x)
        '''(batch_size, time_steps, class_num)'''

        # Interpolate
        azimuth_output = interpolate(azimuth_output, self.interp_ratio)
        azimuth_output = self.last_activation(azimuth_output)

        pred = azimuth_output.mean(1)
        #prediction = scores.max(-1)[1]
        return pred

class CRNNv2(nn.Module):
    def __init__(self, c_in, class_num=10, pool_type='avg', pool_size=(2,2), pretrained_path=None,last_activation="Sigmoid"):
        super().__init__()

        self.class_num = class_num
        self.pool_type = pool_type
        self.pool_size = pool_size
        
        self.conv_block0 = ConvBlock(in_channels=c_in, out_channels=64)    # 1: 7, 128     2: 7, 64
        self.gate_block0 = ConvBlock(in_channels=4, out_channels=64)

        
        self.gru = nn.GRU(input_size=64, hidden_size=64, 
            num_layers=3, dropout=0.3, batch_first=True, bidirectional=True)

        self.fc_1 = nn.Sequential(
            nn.Linear(128, 512, bias=True),
            nn.PReLU()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(512, 128, bias=True),
            nn.PReLU()
        )
        self.fc_3 = nn.Linear(128, class_num, bias=True)

        self.init_weights()

        if last_activation == "Sigmoid" :
            self.last_activation = nn.Sigmoid()
        elif last_activation == "Softmax" : 
            self.last_activation = nn.Softmax()
        else : 
            self.last_activation = nn.Sigmoid()

    def init_weights(self):

        init_gru(self.gru)
        init_layer(self.fc_1)
        init_layer(self.fc_2)
        init_layer(self.fc_3)

    def forward(self, x):
        #pdb.set_trace() 
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        gate = self.gate_block0(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block0(x, self.pool_type, pool_size=self.pool_size)
        x = x * torch.sigmoid(gate)
        
        if self.pool_type == 'avg':
            x = torch.mean(x, dim=2)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=2)
        '''(batch_size, feature_maps, time_steps)'''
        
        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        self.gru.flatten_parameters()
        (x, _) = self.gru(x)
        # 
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        '''(batch_size, time_steps, class_num)'''

        azimuth_output = self.last_activation(x)
        pred = azimuth_output.mean(1)
        
        return pred

class TCRN(nn.Module):
    def __init__(self, c_in, class_num=10, last_activation="Sigmoid"):
        super().__init__()

        self.class_num = class_num

        self.enc_1 = nn.Conv1d(257*4,256,1)
        
        self.enc_2 = TCN(
            c_in = 256,
            c_out = 256,
            TCN_activation="PReLU"
        )

        self.rnn_1 = nn.GRU(
            input_size =256,
            hidden_size = 256,
            num_layers =2,
            dropout =0.3,
            bidirectional=False,
            batch_first=True
        )
        
        self.fc_1 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.PReLU()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(256, class_num, bias=True),
        )


        if last_activation == "Sigmoid" :
            self.last_activation = nn.Sigmoid()
        elif last_activation == "Softmax" : 
            self.last_activation = nn.Softmax()
        else : 
            self.last_activation = nn.Sigmoid()

    def forward(self, x):
        # [B,C,T,F]
        
        # [B, C, F,T]
        x = torch.permute(x,(0,1,3,2))
        
        # [B,C*F,T]
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2],x.shape[3]))
        
        x = self.enc_1(x)
        
        x = self.enc_2(x)
        
        x = torch.permute(x,(0,2,1))
        
        x = self.rnn_1(x)[0]
        
        x = self.fc_1(x)
        
        x = self.fc_2(x)
        
        
        azimuth_output = self.last_activation(x)

        pred = azimuth_output.mean(1)

        return pred

class TCRNv2(nn.Module):
    def __init__(self, c_in, class_num=10, last_activation="Sigmoid"):
        super().__init__()

        self.class_num = class_num

        self.enc_1 = TCN(
            c_in = 1028,
            c_out = 1028,
            TCN_activation="PReLU"
        )
        
        self.enc_2 = TCN(
            c_in = 1028,
            c_out = 1028,
            TCN_activation="PReLU"
        )

        self.rnn_1 = nn.GRU(
            input_size =1028,
            hidden_size = 512,
            num_layers =2,
            dropout =0.3,
            bidirectional=False,
            batch_first=True
        )
        
        self.fc_1 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.PReLU()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(256, class_num, bias=True),
        )


        if last_activation == "Sigmoid" :
            self.last_activation = nn.Sigmoid()
        elif last_activation == "Softmax" : 
            self.last_activation = nn.Softmax()
        else : 
            self.last_activation = nn.Sigmoid()

    def forward(self, x):
        # [B,C,T,F]
        
        # [B, C, F,T]
        x = torch.permute(x,(0,1,3,2))
        
        # [B,C*F,T]
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2],x.shape[3]))
        
        x = self.enc_1(x)
        
        x = self.enc_2(x)
        
        x = torch.permute(x,(0,2,1))
        
        x = self.rnn_1(x)[0]
        
        x = self.fc_1(x)
        
        x = self.fc_2(x)
        
        
        azimuth_output = self.last_activation(x)

        pred = azimuth_output.mean(1)
        #prediction = scores.max(-1)[1]
        return pred

# Test
if __name__ == "__main__" : 

    from utils.hparams import HParam
    hp = HParam("../config/v0.yaml","../config/default.yaml")