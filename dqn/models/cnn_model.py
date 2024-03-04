import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from models.NoisyNet import NoisyLinear
from models.transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class CNN_model(nn.Module):
    #sketchを使わないCNNモデル
    def __init__(self, obs_shape, n_actions, device):
        super(CNN_model, self).__init__()
        noisy_std = 0.1
        hidden_dim = 128

        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            )

        self.pos_ang_linear =nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            )

        self.sketch_linear =nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(200, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            )

        conv_out_size = self.get_conv_out(obs_shape)
        #self.value_linear = nn.Sequential(nn.Linear(conv_out_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.Linear1 = NoisyLinear(conv_out_size+200, hidden_dim, std_init=noisy_std)
        self.Linear2 = NoisyLinear(hidden_dim, n_actions, std_init=noisy_std)

    def get_conv_out(self, obs_shape):
        x = torch.zeros(1, *obs_shape)
        x = self.cnn(x)
        return int(np.prod(x.size()))

    def forward(self, x, sk):
        image, angle, pos_x, pos_y = x
        self.input_image = image * 1.0
        pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)

        x = self.cnn(image)
        x = x.reshape(x.shape[0],-1)

        p = self.pos_ang_linear(pos_ang)
        s = self.sketch_linear(sk.flatten(1))

        out = self.Linear2(F.relu(self.Linear1(torch.cat([x,s,p],dim=1))))
        #out = v.expand(-1, out.size(1)) + out - out.mean(1, keepdim=True).expand(-1, out.size(1))
        return out

    def get_det_action(self, x, sk):
        qs = self.forward(x, sk)
        return torch.argmax(qs, dim=1).unsqueeze(1)
    
    def reset_noise(self):
        for name, module in self.named_children():
            if 'Linear' in name:
                module.reset_noise()

class CNN_SKT_model(nn.Module):
    #sketchを用いたCNNモデル
    def __init__(self, obs_shape, n_actions, device):
        super(CNN_SKT_model, self).__init__()
        hidden_dim = 128
        num_encoder_layers = 1
        noisy_std = 0.1

        self.sketch_encoder = nn.Sequential(nn.Linear(2,hidden_dim))
        self.sketch_pos_embedding = nn.Parameter(torch.randn(100, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.mask = torch.triu(torch.full((100,100), float('-inf'), device=device), diagonal=1).to(torch.bool)

        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            )

        self.pos_ang_linear =nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            )

        conv_out_size = self.get_conv_out(obs_shape)
        #self.value_linear = nn.Sequential(nn.Linear(conv_out_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.Linear1 = NoisyLinear(conv_out_size + hidden_dim + 100, hidden_dim, std_init=noisy_std)
        self.Linear2 = NoisyLinear(hidden_dim, n_actions, std_init=noisy_std)

    def get_conv_out(self, obs_shape):
        x = torch.zeros(1, *obs_shape)
        x = self.cnn(x)
        return int(np.prod(x.size()))
    
    def create_query(self, points):
        # sketch_transformer encoder
        sketch_embed = self.sketch_encoder(points).permute(1,0,2)

        sketch_embed += self.sketch_pos_embedding
        
        sketch_embeds = self.transformer_encoder(sketch_embed, mask=self.mask) # 10,bs,256

        return sketch_embeds

    def forward(self, x, sketchs, t):
        image, angle, pos_x, pos_y = x
        # 時刻情報が1の場合、環境分複製
        t = np.repeat(t,image.shape[0]) if len(t) == 1 else t
        pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)
        # 時刻が100の時99に変更
        t = [99 if i == 100 else i for i in t]
        self.input_image = image * 1.0

        x = self.cnn(image)
        x = x.reshape(x.shape[0],-1)
        #v = self.value_linear(x)

        sketch = []
        for i,time in enumerate(t):
            one_sketch = sketchs[time][i]
            sketch.append(one_sketch)
        sketchs = torch.stack(sketch, dim=0)
        p = self.pos_ang_linear(pos_ang)

        out = self.Linear2(F.relu(self.Linear1(torch.cat([x,sketchs,p],dim=1))))
        #out = v.expand(-1, out.size(1)) + out - out.mean(1, keepdim=True).expand(-1, out.size(1))
        return out

    def get_det_action(self, x, sk, t):
        qs = self.forward(x, sk, t)
        return torch.argmax(qs, dim=1).unsqueeze(1)
    
    def reset_noise(self):
        for name, module in self.named_children():
            if 'Linear' in name:
                module.reset_noise()

class CNN_SKT__nopos_model(nn.Module):
    #sketchを用いたCNNモデル
    def __init__(self, obs_shape, n_actions, device):
        super(CNN_SKT__nopos_model, self).__init__()
        hidden_dim = 128
        num_encoder_layers = 1
        noisy_std = 0.1

        self.sketch_encoder = nn.Sequential(nn.Linear(2,hidden_dim))
        self.sketch_pos_embedding = nn.Parameter(torch.randn(100, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.mask = torch.triu(torch.full((100,100), float('-inf'), device=device), diagonal=1).to(torch.bool)

        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            )

        conv_out_size = self.get_conv_out(obs_shape)
        #self.value_linear = nn.Sequential(nn.Linear(conv_out_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.Linear1 = NoisyLinear(conv_out_size + hidden_dim, hidden_dim, std_init=noisy_std)
        self.Linear2 = NoisyLinear(hidden_dim, n_actions, std_init=noisy_std)

    def get_conv_out(self, obs_shape):
        x = torch.zeros(1, *obs_shape)
        x = self.cnn(x)
        return int(np.prod(x.size()))
    
    def create_query(self, points):
        # sketch_transformer encoder
        sketch_embed = self.sketch_encoder(points).permute(1,0,2)

        sketch_embed += self.sketch_pos_embedding
        
        sketch_embeds = self.transformer_encoder(sketch_embed, mask=self.mask) # 10,bs,256

        return sketch_embeds

    def forward(self, x, sketchs, t):
        image, angle, pos_x, pos_y = x
        # 時刻情報が1の場合、環境分複製
        t = np.repeat(t,image.shape[0]) if len(t) == 1 else t
        #pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)
        # 時刻が100の時99に変更
        t = [99 if i == 100 else i for i in t]
        self.input_image = image * 1.0

        x = self.cnn(image)
        x = x.reshape(x.shape[0],-1)
        #v = self.value_linear(x)

        sketch = []
        for i,time in enumerate(t):
            one_sketch = sketchs[time][i]
            sketch.append(one_sketch)
        sketchs = torch.stack(sketch, dim=0)
        #p = self.pos_ang_linear(pos_ang)

        out = self.Linear2(F.relu(self.Linear1(torch.cat([x,sketchs],dim=1))))
        #out = v.expand(-1, out.size(1)) + out - out.mean(1, keepdim=True).expand(-1, out.size(1))
        return out

    def get_det_action(self, x, sk, t):
        qs = self.forward(x, sk, t)
        return torch.argmax(qs, dim=1).unsqueeze(1)
    
    def reset_noise(self):
        for name, module in self.named_children():
            if 'Linear' in name:
                module.reset_noise()

class CNN_SKT_full_model(nn.Module):
    #sketchを用いたCNNモデル
    def __init__(self, obs_shape, n_actions, device):
        super(CNN_SKT_full_model, self).__init__()
        hidden_dim = 128
        num_encoder_layers = 1
        noisy_std = 0.1

        self.sketch_encoder = nn.Sequential(nn.Linear(2,hidden_dim))
        self.sketch_pos_embedding = nn.Parameter(torch.randn(100, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.mask = torch.triu(torch.full((100,100), float('-inf'), device=device), diagonal=1).to(torch.bool)

        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            )

        self.pos_ang_linear =nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            )

        conv_out_size = self.get_conv_out(obs_shape)
        #self.value_linear = nn.Sequential(nn.Linear(conv_out_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.Linear1 = NoisyLinear(conv_out_size + 100 + hidden_dim, hidden_dim, std_init=noisy_std)
        self.Linear2 = NoisyLinear(hidden_dim, n_actions, std_init=noisy_std)

    def get_conv_out(self, obs_shape):
        x = torch.zeros(1, *obs_shape)
        x = self.cnn(x)
        return int(np.prod(x.size()))
    
    def create_query(self, points):
        # sketch_transformer encoder
        sketch_embed = self.sketch_encoder(points).permute(1,0,2)

        sketch_embed += self.sketch_pos_embedding
        
        sketch_embeds = self.transformer_encoder(sketch_embed, mask=self.mask) # 10,bs,256

        return sketch_embeds

    def forward(self, x, sketchs, t):
        image, angle, pos_x, pos_y = x
        # 時刻情報が1の場合、環境分複製
        t = np.repeat(t,image.shape[0]) if len(t) == 1 else t
        pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)
        # 時刻が100の時99に変更
        t = [99 if i == 100 else i for i in t]
        self.input_image = image * 1.0

        x = self.cnn(image)
        x = x.reshape(x.shape[0],-1)
        #v = self.value_linear(x)
        p = self.pos_ang_linear(pos_ang)

        out = self.Linear2(F.relu(self.Linear1(torch.cat([x,p,sketchs],dim=1))))
        #out = v.expand(-1, out.size(1)) + out - out.mean(1, keepdim=True).expand(-1, out.size(1))
        return out

    def get_det_action(self, x, sk, t):
        qs = self.forward(x, sk, t)
        return torch.argmax(qs, dim=1).unsqueeze(1)
    
    def reset_noise(self):
        for name, module in self.named_children():
            if 'Linear' in name:
                module.reset_noise()