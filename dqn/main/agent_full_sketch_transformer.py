import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from datetime import datetime
import pytz
import psutil
import sys
import random
import os

from models.transformer_model import SKT_full_model
from replay_buffer import replay_buffer
from env import unity_env
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from utils import REWARD_LOGGER


class Agent:
    def __init__(self, args):
        if args.seed != 0:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            random.seed(args.seed)
        self.args = args

        self.device = torch.device(self.args.device)
        self.best_reward = -sys.maxsize
        self.sketch_number = args.sketch_size

        self.args.save_direct = self.args.save_direct + "Full_SKT/"
        if not os.path.exists(self.args.save_direct):
            os.makedirs(self.args.save_direct)

        #### create new log file for each run
        if self.args.load_dir is None:
            #### get number of log files in log directory
            self.run_num = 0
            current_num_dir = next(os.walk(self.args.save_direct))[1]
            self.run_num = len(current_num_dir)

            self.args.save_direct = self.args.save_direct + '/' + str(self.run_num) + '/'
            if not os.path.exists(self.args.save_direct):
                os.makedirs(self.args.save_direct)
                
            self.save_dir = self.args.save_direct + '/models/'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            log_f_name = self.args.save_direct + '/Full_SKT'
            self.log_f = REWARD_LOGGER(log_f_name)

            self.load_dir = self.save_dir
        else:
            #### get number of log files in log directory
            self.run_num = 0
            current_num_dir = next(os.walk(self.args.save_direct))[1]
            self.run_num = len(current_num_dir)
            
            self.args.save_direct = self.args.save_direct + '/' + str(self.run_num) + '/'
            if not os.path.exists(self.args.save_direct):
                os.makedirs(self.args.save_direct)

            log_f_name = self.args.save_direct + '/Full_SKT'
            self.log_f = REWARD_LOGGER(log_f_name, test=True)

            self.save_dir = self.args.save_direct
            self.load_dir = self.args.load_dir
        
        self.env = unity_env(file_name=args.env_name, worker_id=self.run_num, time_scale=args.time_scale, seed=args.seed, device=self.device, num_sketch=self.sketch_number)
        
        self.args.num_envs = self.env.num_envs
        self.n_cycles = self.args.n_cycles//self.env.num_envs
        if self.args.n_cycles%self.env.num_envs!=0:
            sys.exit('cycles no_loop')
        self.buffer = replay_buffer(self.args)
        
        self.epsilon = self.args.epsilon_start
        self.epsilon_decay = self.args.epsilon_decay*self.n_cycles

        self.Q = SKT_full_model(args=self.args, device=self.device).to(self.device)
        self.Q_target = SKT_full_model(args=self.args, device=self.device).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.args.lr)
        self.MSEloss = nn.MSELoss() #nn.SmoothL1Loss()
        self.scaler = torch.cuda.amp.GradScaler() 


    def learn(self):
        start_time = datetime.now()
        print('[' + str(datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y%m%dT%H%M%S")) + '] now start...')
        self.update_count = 0
        self.episode_count = 0
        for epoch in range(self.args.n_epochs):
            self.Q.train()
            for _ in range(self.n_cycles):
                ep_obs    = torch.empty((self.args.num_envs,self.args.ep_len+1,self.args.obs_shape[0],self.args.obs_shape[1],self.args.obs_shape[2]),device=self.device)
                ep_action = torch.empty((self.args.num_envs,self.args.ep_len),device=self.device)
                ep_reward = torch.empty((self.args.num_envs,self.args.ep_len),device=self.device)
                ep_angle = torch.empty((self.args.num_envs,self.args.ep_len+1),device=self.device)
                ep_pos_x = torch.empty((self.args.num_envs,self.args.ep_len+1),device=self.device)
                ep_pos_y = torch.empty((self.args.num_envs,self.args.ep_len+1),device=self.device)
                ep_point = torch.empty((self.args.num_envs,self.args.ep_len,2),device=self.device)

                rand = torch.randint(self.sketch_number, (self.args.num_envs,1),device=self.device) #1
                obs, pattern = self.env.reset(rand)

                sk = self.Q.create_query(pattern)
                for i in range(self.args.ep_len):
                    action = self.Q.get_det_action(obs,sk,[i])
                    next_obs, dis_rew, way_rew = self.env.step(action, i)
                    reward = way_rew # dis_rew + 

                    if i % self.args.replay_frequency == 0:
                        self.Q.reset_noise()  # Draw a new set of noisy weights

                    for j in range(self.args.num_envs):
                        ep_obs[j,i] = obs[0][j][:]
                        ep_angle[j,i] = obs[1][j]
                        ep_pos_x[j,i] = obs[2][j]
                        ep_pos_y[j,i] = obs[3][j]
                        ep_action[j,i] = action[j]
                        ep_reward[j,i] = reward[j]

                    obs = next_obs
                for j in range(self.args.num_envs):
                    ep_obs[j][-1] = next_obs[0][j][:]
                    ep_angle[j][-1] = obs[1][j]
                    ep_pos_x[j][-1] = obs[2][j]
                    ep_pos_y[j][-1] = obs[3][j]
                ep_point = pattern
                self.buffer.append(ep_obs, ep_action, ep_reward, ep_angle, ep_pos_x, ep_pos_y, ep_point)
                self.episode_count += self.args.num_envs
                
                for _ in range(self.args.n_updates):
                    loss=self.update_network()
                    self.update_count+=1
                    now_time = datetime.now()
                    time_taken = now_time - start_time
                    print('\r'+'['+str(time_taken)+'] Full_SKT:' + str(self.run_num) + \
                        ' epoch:'+str(epoch+1) +' episode:'+ str(self.episode_count), end='', flush=True)# + ' episilon:' + str(round(self.epsilon,3))
                #self.update_epsilon()
            torch.save(self.Q.state_dict(), self.save_dir + 'model.pt')
            reward=self.eval_agent()
            self.log_f.record_train_log(self.episode_count, reward.round(decimals=2), loss.item())
            self.log_f._train_log_plot()

        print('done')
        self.env.close()


    """def update_epsilon(self):
        self.epsilon -= (self.args.epsilon_start - self.args.epsilon_end) / self.epsilon_decay
        self.epsilon = max(self.epsilon , self.args.epsilon_end)

    def get_action(self,obs,sk,t,train=True):
        t = [t]
        if train:
            # 時刻情報が1の場合、環境分複製
            t = np.repeat(t,self.args.num_envs) if len(t) == 1 else t
        if random.random() < self.epsilon and train:
            action = torch.randint(self.args.action_size, (self.args.num_envs,1))
        else:
            action = self.Q.get_det_action(obs,sk,t)
        return action"""


    def update_network(self):
        if self.buffer.len() < self.args.start_train:
            return
        obs, next_obs, action, reward, ang, next_ang, px, npx, py, npy, pattern, t = self.buffer.sample()
        obs        = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs   = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        action     = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward     = torch.tensor(reward, dtype=torch.float32, device=self.device)
        ang        = torch.tensor(ang, dtype=torch.float32, device=self.device)
        next_ang   = torch.tensor(next_ang, dtype=torch.float32, device=self.device)
        px         = torch.tensor(px, dtype=torch.float32, device=self.device)
        npx        = torch.tensor(npx, dtype=torch.float32, device=self.device)
        py         = torch.tensor(py, dtype=torch.float32, device=self.device)
        npy        = torch.tensor(npy, dtype=torch.float32, device=self.device)
        pattern    = torch.tensor(pattern, dtype=torch.float32, device=self.device)
        batch_idx  = torch.arange(self.args.batch_size)

        obs = [obs, ang, px, py]
        next_obs = [next_obs, next_ang, npx, npy]

        with torch.cuda.amp.autocast():
            sk = self.Q.create_query(pattern)
            with torch.no_grad():
                sk_target = sk.detach() #self.Q_target.create_query(rand)
                next_action = self.Q(next_obs, sk_target, t+1).max(1)[1]
                self.Q_target.reset_noise()
                next_q = self.Q_target(next_obs, sk_target, t+1)[batch_idx, next_action]

            q_target = next_q * self.args.gamma + reward
            q = self.Q(obs, sk, t)[batch_idx, action]
            q_loss = self.MSEloss(q, q_target)

        self.optimizer.zero_grad()
        self.scaler.scale(q_loss).backward() 
        clip_grad_norm_(self.Q.parameters(),1)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.soft_update(self.Q, self.Q_target)
        #if self.update_count % 100000 == 0:
            #self.Q_target.load_state_dict(self.Q.state_dict())

        if psutil.virtual_memory().percent>95.0:
            sys.exit('memory usage unnormal')

        return q_loss


    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.args.tau*local_param.data + (1.0-self.args.tau)*target_param.data)

    def eval_agent(self,demo=False):
        if not demo:
            self.Q.load_state_dict(torch.load(self.load_dir + 'model.pt'))
            self.Q.eval()
            test_envs = self.args.n_test*self.args.num_envs
            score = 0
            for _ in range(self.args.n_test):
                rand = torch.randint(self.sketch_number, (self.args.num_envs,1),device=self.device) #1
                obs, pattern = self.env.reset(rand)
                sk=self.Q.create_query(pattern)
                for i in range(self.args.ep_len):
                    action = self.Q.get_det_action(obs,sk,[i])
                    next_obs, dis_rew, way_rew = self.env.step(action, i)
                    reward = way_rew # dis_rew + 
                    score += reward
                    obs = next_obs

            avg_score = sum(score)/test_envs
            print(' - avg_score:{:.2f}'.format(avg_score))
            if self.best_reward < avg_score:
                self.best_reward=avg_score
                torch.save(self.Q.state_dict(), self.save_dir + "best_model.pt")
            if self.episode_count % 10000 == 0:
                torch.save(self.Q.state_dict(), self.save_dir + str(self.episode_count) + "_model.pt")
            return avg_score.clone().detach().cpu().numpy()
        else:
            self.Q.load_state_dict(torch.load(self.load_dir))
            self.demo_agent()

    def demo_agent(self):
        from utils import make_en_attention, make_en_img, make_de_attention, make_de_img ,min_max
        self.Q.eval()
        epi_score, epi_way_score, best_score, best_epi, best_way = 0, 0, -sys.maxsize, 0, -sys.maxsize
        for epi in range(self.args.n_test):
            # make directry
            epi_dir = os.path.join(self.save_dir, "epi{}".format(epi+1))
            if not os.path.exists(epi_dir):
                os.makedirs(epi_dir)
            if not os.path.exists(epi_dir + "/raw_img/"):
                os.makedirs(epi_dir + "/raw_img/")
            if not os.path.exists(epi_dir + "/raw_img_point/"):
                os.makedirs(epi_dir + "/raw_img_point/")
            if not os.path.exists(epi_dir + "/encoder/"):
                os.mkdir(epi_dir + "/encoder/")
            if not os.path.exists(epi_dir + "/decoder/"):
                os.mkdir(epi_dir + "/decoder/")
            for i in range(self.args.action_size):#self.args.ep_len
                if not os.path.exists(epi_dir + "/decoder/decoder_act{}/".format(i)):
                    os.mkdir(epi_dir + "/decoder/decoder_act{}/".format(i))

            raw_list, en_list, de_list, T_Qs = [], [], [], []
            rand = torch.randint(self.sketch_number, (self.args.num_envs,1))
            obs, pattern = self.env.reset(rand)
            score, way_score = 0, 0
            sk = self.Q.create_query(pattern)
            for i in range(self.args.ep_len):

                action, q_val, raw_img, en_atts, de_atts = self.demo_action(obs, sk, i)

                raw_img = raw_img[0].permute(1, 2, 0).cpu().detach().numpy()
                raw_img = raw_img*255
                raw_img = raw_img.astype(np.uint8)
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                raw_list.append(raw_img)

                en_attention = make_en_attention(en_atts, self.device)
                en_list.append(en_attention)

                de_attention = make_de_attention(de_atts, self.args.action_size, self.device)
                de_list.append(de_attention)

                T_Qs.append(q_val[0])

                obs, dis_rew, way_rew = self.env.step(action, i)
                way_score += way_rew[0]
            score += dis_rew[0]
            if score > best_score:
                best_way = way_score
                best_score = score
                best_epi = epi+1
            epi_score += score
            epi_way_score += way_score
            print('{} score:{:.2f} sketch:{} way_score:{:.2f}'.format(epi+1, score, rand[0].item(), way_score))
            
            # make img
            pattern_point = pattern.cpu().detach().numpy()
            pattern_point = pattern_point.astype(np.uint8)
            cv2.imwrite(epi_dir + "/raw_img/raw_{0:06d}.png".format(0), raw_list[0])
            cv2.circle(raw_list[0], (pattern_point[0,0,0],pattern_point[0,0,1]), 1, (0,0,0), -1)
            cv2.imwrite(epi_dir + "/raw_img_point/raw_{0:06d}.png".format(0), raw_list[0])
            raw_list, en_list, de_list = np.array(raw_list), np.array(en_list), np.array(de_list)
            en_max, en_min = en_list.reshape(-1).max(), en_list.reshape(-1).min()
            de_max, de_min = de_list.reshape(-1).max(), de_list.reshape(-1).min()

            """en_mean = np.zeros((en_list.shape[1], en_list.shape[2]))
            for idx in range(en_list.shape[0]):
                en_mean += en_list[idx]
            en_max = max(en_mean.flatten())
            en_min = min(en_mean.flatten())
            for x in range(en_mean.shape[0]):
                for y in range(en_mean.shape[1]):
                    #en_mean[x][y] = en_mean[x][y] / en_list.shape[0]
                    en_mean[x][y] = en_mean[x][y] / en_max
            #en_mean = min_max(en_mean, en_min, en_max)
            
            #en_mean = np.mean(en_list, axis=0)
            en_max, en_min = en_mean.reshape(-1).max(), en_mean.reshape(-1).min()
            en_mean = min_max(en_mean, en_min, en_max)
            make_en_img(en_mean * 255, raw_list[0], 0, epi_dir, mode="mean")"""

            for idx in tqdm(range(len(raw_list))):
                q_val = T_Qs[idx]
                raw_img = raw_list[idx]
                en_att = en_list[idx]
                en_att = min_max(en_att, en_min, en_max)
                de_att = de_list[idx]
                de_att = min_max(de_att, de_min, de_max)

                cv2.imwrite(epi_dir + "/raw_img/raw_{0:06d}.png".format(idx), raw_img)
                cv2.circle(raw_img, (pattern_point[0,idx,0],pattern_point[0,idx,1]), 1, (0,0,0), -1)
                cv2.imwrite(epi_dir + "/raw_img_point/raw_{0:06d}.png".format(idx), raw_img)
                make_en_img(en_att * 255, raw_img, idx, epi_dir)
                make_de_img(q_val, de_att * 255, idx, self.args.action_size, epi_dir)#

        avg_score = epi_score/self.args.n_test
        avg_way_score = epi_way_score/self.args.n_test
        print('\n avg_score:{:.2f} avg_way_score:{:.2f} \n run_num:{} best_epi:{} best_score{:.2f} best_way_score{:.2f}'.format(avg_score, avg_way_score, self.run_num, best_epi, best_score, best_way))
        self.log_f.record_eval_log(1, avg_score.clone().detach().cpu().numpy().round(decimals=2))
        self.env.close()
    
    def demo_action(self,obs,sk,t):
        img_enc_attn_weights, skt_dec_attn_weights = [], []
        hooks = [
          self.Q.image_transformer_encoder.layers[-1].self_attn.register_forward_hook(
              lambda self, input, output: img_enc_attn_weights.append(output[1])
          ),
          self.Q.transformer_decoder.layers[-1].multihead_attn.register_forward_hook(
              lambda self, input, output: skt_dec_attn_weights.append(output[1])
          ),
        ]

        q = self.Q(obs,sk,[t])
        action = torch.argmax(q, dim=1).unsqueeze(1)#.item()

        for hook in hooks:
            hook.remove()
        raw_img = self.Q.input_image

        enc_attn_weights = img_enc_attn_weights[0]
        dec_attn_weights = skt_dec_attn_weights[0]

        return action, q, raw_img, enc_attn_weights, dec_attn_weights