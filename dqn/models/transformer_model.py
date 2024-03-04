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

class SKT_model(nn.Module):
    def __init__(self, args, device):
        super(SKT_model, self).__init__()

        self.args = args

        hidden_dim = 128
        num_encoder_layers = 1
        num_decoder_layers = 1
        patch_size = 32
        noisy_std = 0.1

        self.sketch_embedding = nn.Linear(2,hidden_dim)
        self.sketch_pos_embedding = nn.Parameter(torch.randn(100, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        image_height, image_width = pair(self.args.obs_shape[1])#copy the number
        patch_height, patch_width = pair(patch_size)#copy the number
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.' 

        num_patches = (image_height // patch_height) * (image_width // patch_width)#how many patches
        patch_dim = self.args.obs_shape[0] * patch_height * patch_width#patch image to vector

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), #(h,w) patch number
            nn.Linear(patch_dim, hidden_dim),
            Rearrange('b n d -> n b d')
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.image_transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.pos_ang_linear = nn.Linear(3, hidden_dim)
        
        #self.value_linear = nn.Sequential(nn.Linear(num_patches*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.Linear1 = NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std)
        self.Linear2 = NoisyLinear(hidden_dim, 1, std_init=noisy_std)

        self.act_list = torch.zeros(self.args.action_size, self.args.action_size, device=device)
        for i in range(self.args.action_size):
            self.act_list[i][i] = 1.0
        self.action_encoder = nn.Linear(self.args.action_size+3+hidden_dim, hidden_dim)

        self.mask = torch.triu(torch.full((100,100), float('-inf'), device=device), diagonal=1).to(torch.bool)
    
    def create_query(self, pattern):
        # sketch_transformer encoder
        sketch_token = self.sketch_embedding(pattern).permute(1,0,2)

        sketch_token += self.sketch_pos_embedding
        
        sketch_embed = self.transformer_encoder(sketch_token, mask=self.mask) # 100,4,1

        return sketch_embed
    
    def forward(self, x, sketchs, t):
        image, angle, pos_x, pos_y = x
        # 時刻情報が1の場合、環境分複製
        t = np.repeat(t,image.shape[0]) if len(t) == 1 else t
        pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)
        # 時刻が100の時99に変更
        t = [99 if i == 100 else i for i in t]
        self.input_image = image * 1.0

        src = self.to_patch_embedding(image)

        n, bs, p = src.shape
        src += self.pos_embedding

        memory = self.image_transformer_encoder(src)
        #v = self.value_linear(memory.permute(1,0,2).flatten(1))

        # sketchを時刻部分にスライス
        sketch = []
        for i,time in enumerate(t):
            one_sketch = sketchs[time][i]
            sketch.append(one_sketch)
        sketchs = torch.stack(sketch, dim=0)

        # action queries
        query_embeds = []
        act_lists = self.act_list.unsqueeze(0).repeat(bs, 1, 1)
        for act_list, pos, skt in zip(act_lists,pos_ang, sketchs):
            query_embed = []
            for action in act_list:
                action = torch.cat([action,pos,skt])
                action_query = self.action_encoder(action)
                query_embed.append(action_query)
            query_embed = torch.stack(query_embed, dim=0)
            query_embeds.append(query_embed)
        tgt = torch.stack(query_embeds, dim=1)

        out = self.transformer_decoder(tgt, memory)[0].permute(1, 0, 2) #, tgt_mask=self.mask

        out = self.Linear2(F.relu(self.Linear1(out))).permute(2, 0, 1).squeeze(0)
        #out = v.expand(-1, out.size(1)) + out - out.mean(1, keepdim=True).expand(-1, out.size(1))
        return out

    def get_det_action(self, x, sk, t):
        qs = self.forward(x, sk, t)
        return torch.argmax(qs, dim=1).unsqueeze(1)
    
    def reset_noise(self):
        for name, module in self.named_children():
            if 'Linear' in name:
                module.reset_noise()

    
class AQT_model(nn.Module):
    #sketchを使わないAQTモデル
    def __init__(self, obs_shape, n_actions, device):
        super(AQT_model, self).__init__()

        hidden_dim = 128
        num_encoder_layers = 1
        num_decoder_layers = 1
        patch_size = 32
        noisy_std = 0.1

        image_height, image_width = pair(obs_shape[1])#copy the number
        patch_height, patch_width = pair(patch_size)#copy the number
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.' 

        num_patches = (image_height // patch_height) * (image_width // patch_width)#how many patches
        patch_dim = obs_shape[0] * patch_height * patch_width#patch image to vector

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), #(h,w) patch number
            nn.Linear(patch_dim, hidden_dim),
            Rearrange('b n d -> n b d')
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.image_transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        #self.value_linear = nn.Sequential(nn.Linear(num_patches*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.Linear1 = NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std)
        self.Linear2 = NoisyLinear(hidden_dim, 1, std_init=noisy_std)

        self.act_list = torch.zeros(n_actions, n_actions, device=device)
        for i in range(n_actions):
            self.act_list[i][i] = 1.0
        self.action_encoder = nn.Linear(n_actions+3, hidden_dim)
    
    def forward(self, x):
        image, angle, pos_x, pos_y = x
        pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)
        self.input_image = image * 1.0

        src = self.to_patch_embedding(image)

        n, bs, p = src.shape
        src += self.pos_embedding

        memory = self.image_transformer_encoder(src)
        #v = self.value_linear(memory.permute(1,0,2).flatten(1))

        # action queries
        query_embeds = []
        act_lists = self.act_list.unsqueeze(0).repeat(bs, 1, 1)
        for act_list, pos in zip(act_lists,pos_ang):
            query_embed = []
            for action in act_list:
                action = torch.cat([action,pos])
                action_query = self.action_encoder(action)
                query_embed.append(action_query)
            query_embed = torch.stack(query_embed, dim=0)
            query_embeds.append(query_embed)
        tgt = torch.stack(query_embeds, dim=1)

        out = self.transformer_decoder(tgt, memory)[0]
        out = out.permute(1, 0, 2)

        out = self.Linear2(F.relu(self.Linear1(out))).permute(2, 0, 1).squeeze(0)
        #out = v.expand(-1, out.size(1)) + out - out.mean(1, keepdim=True).expand(-1, out.size(1))
        return out

    def get_det_action(self, x):
        qs = self.forward(x)
        return torch.argmax(qs, dim=1).unsqueeze(1)
    
    def reset_noise(self):
        for name, module in self.named_children():
            if 'Linear' in name:
                module.reset_noise()

    
class AQT_skt_model(nn.Module):
    #sketchを全部使うAQTモデル
    def __init__(self, obs_shape, n_actions, device):
        super(AQT_skt_model, self).__init__()

        hidden_dim = 128
        num_encoder_layers = 1
        num_decoder_layers = 1
        patch_size = 16
        noisy_std = 0.1

        image_height, image_width = pair(obs_shape[1])#copy the number
        patch_height, patch_width = pair(patch_size)#copy the number
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.' 

        num_patches = (image_height // patch_height) * (image_width // patch_width)#how many patches
        patch_dim = obs_shape[0] * patch_height * patch_width#patch image to vector

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), #(h,w) patch number
            nn.Linear(patch_dim, hidden_dim),
            Rearrange('b n d -> n b d')
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.image_transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        #self.value_linear = nn.Sequential(nn.Linear(num_patches*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.sketch_linear =nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(200, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            )

        self.Linear1 = NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std)
        self.Linear2 = NoisyLinear(hidden_dim, 1, std_init=noisy_std)

        self.act_list = torch.zeros(n_actions, n_actions, device=device)
        for i in range(n_actions):
            self.act_list[i][i] = 1.0
        self.action_encoder = nn.Linear(n_actions+103, hidden_dim)
    
    def forward(self, x, sk):
        image, angle, pos_x, pos_y = x
        pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)
        self.input_image = image * 1.0

        src = self.to_patch_embedding(image)

        n, bs, p = src.shape
        src += self.pos_embedding

        memory = self.image_transformer_encoder(src)
        #v = self.value_linear(memory.permute(1,0,2).flatten(1))

        s = self.sketch_linear(sk.flatten(1))

        # action queries
        query_embeds = []
        act_lists = self.act_list.unsqueeze(0).repeat(bs, 1, 1)
        for act_list, pos, skt in zip(act_lists,pos_ang, s):
            query_embed = []
            for action in act_list:
                action = torch.cat([action,pos,skt])
                action_query = self.action_encoder(action)
                query_embed.append(action_query)
            query_embed = torch.stack(query_embed, dim=0)
            query_embeds.append(query_embed)
        tgt = torch.stack(query_embeds, dim=1)

        out = self.transformer_decoder(tgt, memory)[0]
        out = out.permute(1, 0, 2)

        out = self.Linear2(F.relu(self.Linear1(out))).permute(2, 0, 1).squeeze(0)
        #out = v.expand(-1, out.size(1)) + out - out.mean(1, keepdim=True).expand(-1, out.size(1))
        return out

    def get_det_action(self, x, sk):
        qs = self.forward(x, sk)
        return torch.argmax(qs, dim=1).unsqueeze(1)
    
    def reset_noise(self):
        for name, module in self.named_children():
            if 'Linear' in name:
                module.reset_noise()

class SKT_full_model(nn.Module):
    def __init__(self, args, device):
        super(SKT_full_model, self).__init__()

        self.args = args

        hidden_dim = 128
        num_encoder_layers = 1
        num_decoder_layers = 1
        patch_size = 16
        noisy_std = 0.1

        self.sketch_embedding = nn.Linear(2,hidden_dim)
        self.sketch_pos_embedding = nn.Parameter(torch.randn(100, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        image_height, image_width = pair(self.args.obs_shape[1])#copy the number
        patch_height, patch_width = pair(patch_size)#copy the number
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.' 

        num_patches = (image_height // patch_height) * (image_width // patch_width)#how many patches
        patch_dim = self.args.obs_shape[0] * patch_height * patch_width#patch image to vector

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), #(h,w) patch number
            nn.Linear(patch_dim, hidden_dim),
            Rearrange('b n d -> n b d')
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.image_transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        #self.value_linear = nn.Sequential(nn.Linear(num_patches*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.Linear1 = NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std)
        self.Linear2 = NoisyLinear(hidden_dim, 1, std_init=noisy_std)

        self.act_list = torch.zeros(self.args.action_size, self.args.action_size, device=device)
        for i in range(self.args.action_size):
            self.act_list[i][i] = 1.0
        self.action_encoder = nn.Linear(self.args.action_size+3, hidden_dim)

        self.mask = torch.triu(torch.full((100,100), float('-inf'), device=device), diagonal=1).to(torch.bool)
        self.tgt_mask = torch.triu(torch.full((104,104), float('-inf'), device=device), diagonal=5).to(torch.bool)
    
    def create_query(self, pattern):
        # sketch_transformer encoder
        sketch_token = self.sketch_embedding(pattern).permute(1,0,2)

        sketch_token += self.sketch_pos_embedding
        
        sketch_embed = self.transformer_encoder(sketch_token, mask=self.mask) # 100,4,1

        return sketch_embed
    
    def forward(self, x, sketchs, t):
        image, angle, pos_x, pos_y = x
        pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)
        # 時刻が100の時99に変更
        t = np.repeat(t,self.args.num_envs) if len(t) == 1 else t
        t = [100 if i == 100 else i+1 for i in t]
        self.input_image = image * 1.0

        src = self.to_patch_embedding(image)

        n, bs, p = src.shape
        src += self.pos_embedding

        memory = self.image_transformer_encoder(src)
        #v = self.value_linear(memory.permute(1,0,2).flatten(1))　#dueling_networkの残骸

        # sketchのsizeを1にし、時刻部分にスライス
        tgt_mask = []
        for time in t:
            one_mask = self.tgt_mask[time]
            tgt_mask.append(one_mask)
        tgt_masks = torch.stack(tgt_mask, dim=0)

        # action queries
        query_embeds = []
        act_lists = self.act_list.unsqueeze(0).repeat(bs, 1, 1)
        for act_list, pos in zip(act_lists,pos_ang):
            query_embed = []
            for action in act_list:
                action = torch.cat([action,pos])
                action_query = self.action_encoder(action)
                query_embed.append(action_query)
            query_embed = torch.stack(query_embed, dim=0)
            query_embeds.append(query_embed)
        tgt = torch.cat([torch.stack(query_embeds, dim=1), sketchs])

        out = self.transformer_decoder(tgt, memory, tgt_key_padding_mask=tgt_masks)[0] # (self.args.action_size+100, batch, hidden_dim)
        out = out[:self.args.action_size].permute(1,0,2) # (self.args.action_size+100, batch, hidden_dim) --> (batch, self.args.action_size, hidden_dim)

        out = self.Linear2(F.relu(self.Linear1(out))).permute(2, 0, 1).squeeze(0)
        #out = v.expand(-1, out.size(1)) + out - out.mean(1, keepdim=True).expand(-1, out.size(1))　#dueling_networkの残骸
        return out

    def get_det_action(self, x, sk, t):
        qs = self.forward(x, sk, t)
        return torch.argmax(qs, dim=1).unsqueeze(1)
    
    def reset_noise(self):
        for name, module in self.named_children():
            if 'Linear' in name:
                module.reset_noise()

class SKT_nopos_model(nn.Module):
    def __init__(self, args, device):
        super(SKT_nopos_model, self).__init__()

        self.args = args

        hidden_dim = 128
        num_encoder_layers = 1
        num_decoder_layers = 1
        patch_size = 16
        noisy_std = 0.1

        self.sketch_embedding = nn.Linear(2,hidden_dim)
        self.sketch_pos_embedding = nn.Parameter(torch.randn(100, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        image_height, image_width = pair(self.args.obs_shape[1])#copy the number
        patch_height, patch_width = pair(patch_size)#copy the number
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.' 

        num_patches = (image_height // patch_height) * (image_width // patch_width)#how many patches
        patch_dim = self.args.obs_shape[0] * patch_height * patch_width#patch image to vector

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), #(h,w) patch number
            nn.Linear(patch_dim, hidden_dim),
            Rearrange('b n d -> n b d')
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.image_transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.pos_ang_linear = nn.Linear(3, hidden_dim)
        
        #self.value_linear = nn.Sequential(nn.Linear(num_patches*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.Linear1 = NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std)
        self.Linear2 = NoisyLinear(hidden_dim, 1, std_init=noisy_std)

        self.act_list = torch.zeros(self.args.action_size, self.args.action_size, device=device)
        for i in range(self.args.action_size):
            self.act_list[i][i] = 1.0
        self.action_encoder = nn.Linear(self.args.action_size+hidden_dim, hidden_dim)

        self.mask = torch.triu(torch.full((100,100), float('-inf'), device=device), diagonal=1).to(torch.bool)
    
    def create_query(self, pattern):
        # sketch_transformer encoder
        sketch_token = self.sketch_embedding(pattern).permute(1,0,2)

        sketch_token += self.sketch_pos_embedding
        
        sketch_embed = self.transformer_encoder(sketch_token, mask=self.mask) # 100,4,1

        return sketch_embed
    
    def forward(self, x, sketchs, t):
        image, angle, pos_x, pos_y = x
        # 時刻情報が1の場合、環境分複製
        t = np.repeat(t,image.shape[0]) if len(t) == 1 else t
        #pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)
        # 時刻が100の時99に変更
        t = [99 if i == 100 else i for i in t]
        self.input_image = image * 1.0

        src = self.to_patch_embedding(image)

        n, bs, p = src.shape
        src += self.pos_embedding

        memory = self.image_transformer_encoder(src)
        #v = self.value_linear(memory.permute(1,0,2).flatten(1))

        # sketchを時刻部分にスライス
        sketch = []
        for i,time in enumerate(t):
            one_sketch = sketchs[time][i]
            sketch.append(one_sketch)
        sketchs = torch.stack(sketch, dim=0)

        # action queries
        query_embeds = []
        act_lists = self.act_list.unsqueeze(0).repeat(bs, 1, 1)
        for act_list, skt in zip(act_lists, sketchs):
            query_embed = []
            for action in act_list:
                action = torch.cat([action,skt])
                action_query = self.action_encoder(action)
                query_embed.append(action_query)
            query_embed = torch.stack(query_embed, dim=0)
            query_embeds.append(query_embed)
        tgt = torch.stack(query_embeds, dim=1)

        out = self.transformer_decoder(tgt, memory)[0].permute(1, 0, 2) #, tgt_mask=self.mask

        out = self.Linear2(F.relu(self.Linear1(out))).permute(2, 0, 1).squeeze(0)
        #out = v.expand(-1, out.size(1)) + out - out.mean(1, keepdim=True).expand(-1, out.size(1))
        return out

    def get_det_action(self, x, sk, t):
        qs = self.forward(x, sk, t)
        return torch.argmax(qs, dim=1).unsqueeze(1)
    
    def reset_noise(self):
        for name, module in self.named_children():
            if 'Linear' in name:
                module.reset_noise()
    
class SKT_new_model(nn.Module):
    #sketchを用いたAQTモデル
    def __init__(self, args, device):
        super(SKT_new_model, self).__init__()

        self.args = args

        hidden_dim = 128
        num_encoder_layers = 1
        num_decoder_layers = 1
        patch_size = 16

        self.sketch_embedding_1 = nn.Linear(2,hidden_dim)
        self.sketch_pos_embedding = nn.Parameter(torch.randn(100, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.sketch_embedding_2 = nn.Linear(hidden_dim,1)

        image_height, image_width = pair(self.args.obs_shape[1])#copy the number
        patch_height, patch_width = pair(patch_size)#copy the number
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.' 

        num_patches = (image_height // patch_height) * (image_width // patch_width)#how many patches
        patch_dim = self.args.obs_shape[0] * patch_height * patch_width#patch image to vector

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), #(h,w) patch number
            nn.Linear(patch_dim, hidden_dim),
            Rearrange('b n d -> n b d')
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.image_transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.pos_ang_linear = nn.Linear(3, hidden_dim)
        
        #self.value_linear = nn.Sequential(nn.Linear(num_patches*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.mlp = nn.Linear(hidden_dim, 1)

        self.act_list = torch.zeros(self.args.action_size, self.args.action_size, device=device)
        for i in range(self.args.action_size):
            self.act_list[i][i] = 1.0
        self.action_encoder = nn.Linear(self.args.action_size+4, hidden_dim)

        self.mask = torch.triu(torch.full((100,100), float('-inf'), device=device), diagonal=1).to(torch.bool)
        self.tgt_mask = torch.triu(torch.full((104,104), float('-inf'), device=device), diagonal=5).to(torch.bool)
    
    def create_query(self, pattern):
        # sketch_transformer encoder
        sketch_token = self.sketch_embedding_1(pattern).permute(1,0,2)

        sketch_token += self.sketch_pos_embedding
        
        sketch_embed = self.transformer_encoder(sketch_token, mask=self.mask) # 100,4,1

        #sketch_embed = self.sketch_embedding_2(sketch_embed)

        return sketch_embed
    
    def forward(self, x, sketchs, t):
        image, angle, pos_x, pos_y = x
        pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)
        # 時刻が100の時99に変更
        t = [100 if i == 100 else i+1 for i in t]
        self.input_image = image * 1.0

        src = self.to_patch_embedding(image)

        n, bs, p = src.shape
        src += self.pos_embedding

        memory = self.image_transformer_encoder(src)
        #v = self.value_linear(memory.permute(1,0,2).flatten(1))　#dueling_networkの残骸

        # sketchのsizeを1にし、時刻部分にスライス
        tgt_mask = []
        for time in t:
            one_mask = self.tgt_mask[:self.args.action_size+time]
            tgt_mask.append(one_mask)
        tgt_masks = torch.stack(tgt_mask, dim=0)

        # action queries
        query_embeds = []
        act_lists = self.act_list.unsqueeze(0).repeat(bs, 1, 1)
        for act_list, pos in zip(act_lists,pos_ang):
            query_embed = []
            for action in act_list:
                action = torch.cat([action,pos])
                action_query = self.action_encoder(action)
                query_embed.append(action_query)
            query_embed = torch.stack(query_embed, dim=0)
            query_embeds.append(query_embed)
        tgt = torch.cat([torch.stack(query_embeds, dim=1), sketchs])

        out = self.transformer_decoder(tgt, memory, tgt_key_padding_mask=tgt_masks)[0] # (self.args.action_size+100, batch, hidden_dim)
        out = out[:self.args.action_size].permute(1,0,2) # (self.args.action_size+100, batch, hidden_dim) --> (batch, self.args.action_size, hidden_dim)

        out = self.Linear2(F.relu(self.Linear1(out))).permute(2, 0, 1).squeeze(0)
        #out = v.expand(-1, out.size(1)) + out - out.mean(1, keepdim=True).expand(-1, out.size(1))　#dueling_networkの残骸
        return out

    def get_det_action(self, x, sk, t):
        qs = self.forward(x, sk, t)
        return torch.argmax(qs, dim=1).unsqueeze(1)
    
class VIT_model(nn.Module):
    def __init__(self, obs_shape, n_actions, device):
        super(VIT_model, self).__init__()

        hidden_dim = 128
        num_encoder_layers = 1
        patch_size = 16

        image_height, image_width = pair(obs_shape[1])#copy the number
        patch_height, patch_width = pair(patch_size)#copy the number
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.' 

        num_patches = (image_height // patch_height) * (image_width // patch_width)#how many patches
        patch_dim = obs_shape[0] * patch_height * patch_width#patch image to vector

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), #(h,w) patch number
            nn.Linear(patch_dim, hidden_dim),
            Rearrange('b n d -> n b d')
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.image_transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        

        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(64*hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            )

    def forward(self, x, t):
        src = self.to_patch_embedding(x)
        src += self.pos_embedding
        x = self.image_transformer_encoder(src)
        x = x.permute(1,0,2).flatten(1,2)
        x = self.mlp(x)
        return x

    def get_det_action(self, x, t):
        qs = self.forward(x, t)
        return torch.argmax(qs).item()

class SKT_cat_model(nn.Module):
    def __init__(self, args, device):
        super(SKT_cat_model, self).__init__()

        self.args = args

        hidden_dim = 128
        num_encoder_layers = 1
        num_decoder_layers = 1
        patch_size = 16

        self.sketch_token = _sketch(device)

        self.sketch_embedding_1 = nn.Linear(self.sketch_token.shape[2],hidden_dim)
        self.sketch_pos_embedding = nn.Parameter(torch.randn(self.sketch_token.shape[1], 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        query_encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.transformer_query_encoder = TransformerEncoder(query_encoder_layer, num_encoder_layers)

        self.sketch_embedding_2 = nn.Linear(hidden_dim,1)

        image_height, image_width = pair(self.args.obs_shape[1])#copy the number
        patch_height, patch_width = pair(patch_size)#copy the number
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.' 

        num_patches = (image_height // patch_height) * (image_width // patch_width)#how many patches
        patch_dim = self.args.obs_shape[0] * patch_height * patch_width#patch image to vector

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), #(h,w) patch number
            nn.Linear(patch_dim, hidden_dim),
            Rearrange('b n d -> n b d')
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.image_transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.pos_ang_linear = nn.Linear(3, hidden_dim)
        
        self.mlp = nn.Linear(hidden_dim, 1)

        self.act_list = torch.zeros(self.args.action_size, self.args.action_size, device=device)
        for i in range(self.args.action_size):
            self.act_list[i][i] = 1.0
        self.action_encoder = nn.Linear(self.args.action_size, hidden_dim)

        sk_dim = self.sketch_token.shape[1]
        tgt_dim = self.sketch_token.shape[1]+self.args.action_size+1
        self.mask = torch.triu(torch.full((sk_dim,sk_dim), float('-inf'), device=device), diagonal=1).to(torch.bool)
        self.tgt_mask = torch.triu(torch.full((tgt_dim,tgt_dim), float('-inf'), device=device), diagonal=6).to(torch.bool)
    
    def create_query(self, rand):
        sketch_embeds = []
        for f in rand:
            sketch_embeds.append(self.sketch_token[f.item()][:])
        sketch_embed = torch.stack(sketch_embeds, dim=0) # 100,bs,1

        # sketch_transformer encoder
        sketch_token = self.sketch_embedding_1(sketch_embed).permute(1,0,2)

        sketch_token += self.sketch_pos_embedding
        
        sketch_embed = self.transformer_encoder(sketch_token, mask=self.mask) # 100,4,1

        return sketch_embed
    
    def forward(self, x, sketchs, t):
        image, angle, pos_x, pos_y = x
        pos_ang = torch.stack([angle, pos_x, pos_y], dim=1)
        t = np.repeat(t,self.args.num_envs) if len(t) == 1 else t
        t = [99 if i == 100 else i for i in t]
        self.input_image = image * 1.0

        src = self.to_patch_embedding(image)

        n, bs, p = src.shape
        src += self.pos_embedding

        memory = self.image_transformer_encoder(src)

        pos_ang_embed = self.pos_ang_linear(pos_ang).unsqueeze(0)
        #memory = torch.cat([memory, pos_ang_embed], dim=0)

        """sketch = []
        for i,time in enumerate(t):
            one_sketch = sketchs[time][i]
            sketch.append(one_sketch)
        sketchs = torch.stack(sketch, dim=0)"""

        # action queries
        query_embed = []
        for action in self.act_list:
            action_query = self.action_encoder(action)
            query_embed.append(action_query)
        query_embed = torch.stack(query_embed, dim=0).unsqueeze(1).repeat(1, bs, 1)

        embed = torch.cat([query_embed, pos_ang_embed, sketchs], dim=0)

        embed = self.transformer_query_encoder(embed, mask=self.tgt_mask)
        embed = embed[:4]

        out = self.transformer_decoder(embed, memory)[0] #, tgt_mask=self.mask

        out = self.mlp(out.permute(1, 0, 2)).permute(2, 0, 1).squeeze(0)
        return out

    def get_det_action(self, x, sk, t):
        qs = self.forward(x, sk, t)
        return torch.argmax(qs, dim=1).unsqueeze(1)