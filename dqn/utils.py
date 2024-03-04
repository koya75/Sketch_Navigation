import csv
import os
import matplotlib.pyplot as plt

class REWARD_LOGGER:
    def __init__(self, path, test=False, super=False):
        self.save_path = path
        self.train_record_reward = []
        self.train_record_step = []
        self.test_record_reward = []
        self.test_record_step = []
        
        if test:
            header = ['Episode', 'FDE', "ADE"]
            if not os.path.exists(self.save_path + '_test_log.csv'):
                with open(self.save_path + '_test_log.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
        elif super:
            header = ['Episode', 'Reward', 'Loss', 'Q_Loss', 'TTQ_Loss']
            if not os.path.exists(self.save_path+'_train_log.csv'):
                with open(self.save_path + '_train_log.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
        else:
            header = ['Episode', 'Reward', 'Loss']
            if not os.path.exists(self.save_path+'_train_log.csv'):
                with open(self.save_path + '_train_log.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
    
    def record_train_log(self, epi, reward, loss):
        with open(self.save_path+'_train_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epi, reward, loss])

    def record_train_ttq_log(self, epi, reward, loss, q_loss, ttq_loss):
        with open(self.save_path+'_train_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epi, reward, loss, q_loss, ttq_loss])
        
    def record_eval_log(self, epi, ade, fde):
        with open(self.save_path+'_test_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epi, ade, fde])

    def _train_log_plot(self):
        reader = np.loadtxt(self.save_path+'_train_log.csv', delimiter=',', skiprows=1, unpack=True)
        episode = reader[0]
        reward = reader[1]
        loss = reader[2]

        data01_axis1 = episode
        data01_rewards = reward
        data01_loss = loss
        fig_1 = plt.figure(figsize=(12, 6))
        ax_1 = fig_1.add_subplot(111)
        ax_1.plot(data01_axis1, data01_rewards,  color="k", label="score")
        ax_1.set_xlabel("episode")
        ax_1.set_ylabel("reward")
        ax_1.legend(loc="upper left")
        plt.savefig(self.save_path+'_reward_graph.png', dpi=300)
        plt.close()
        fig_2 = plt.figure(figsize=(12, 6))
        ax_2 = fig_2.add_subplot(111)
        ax_2.plot(data01_axis1, data01_loss,  color="r", label="loss")
        ax_2.set_xlabel("episode")
        ax_2.set_ylabel("loss")
        ax_2.legend(loc="upper left")
        plt.savefig(self.save_path+'_loss_graph.png', dpi=300)
        plt.close()

    def _train_log_ttq_plot(self):
        reader = np.loadtxt(self.save_path+'_train_log.csv', delimiter=',', skiprows=1, unpack=True)
        episode = reader[0]
        reward = reader[1]
        loss = reader[2]
        q_loss = reader[3]
        ttq_loss = reader[4]

        data01_axis1 = episode
        data01_rewards = reward
        data01_loss = loss
        data01_q_loss = q_loss
        data01_ttq_loss = ttq_loss
        fig_1 = plt.figure(figsize=(12, 6))
        ax_1 = fig_1.add_subplot(111)
        ax_1.plot(data01_axis1, data01_rewards,  color="k", label="score")
        ax_1.set_xlabel("episode")
        ax_1.set_ylabel("reward")
        ax_1.legend(loc="upper left")
        plt.savefig(self.save_path+'_reward_graph.png', dpi=300)
        plt.close()
        fig_2 = plt.figure(figsize=(12, 6))
        ax_2 = fig_2.add_subplot(111)
        ax_2.plot(data01_axis1, data01_loss,  color="r", label="loss")
        ax_2.plot(data01_axis1, data01_q_loss,  color="b", label="q_loss")
        ax_2.plot(data01_axis1, data01_ttq_loss,  color="g", label="ttq_loss")
        ax_2.set_xlabel("episode")
        ax_2.set_ylabel("loss")
        ax_2.legend(loc="upper left")
        plt.savefig(self.save_path+'_loss_graph.png', dpi=300)
        plt.close()

    def _test_log_plot(self):
        reader = np.loadtxt(self.save_path+'_test_log.csv', delimiter=',', skiprows=1, unpack=True)
        reward = reader[1]#[row[1] for row in l[1:]]
        episode = reader[0]#[row[2] for row in l[1:]]

        data01_axis1 = episode
        data01_rewards = reward
        fig_1 = plt.figure(figsize=(12, 6))
        ax_1 = fig_1.add_subplot(111)
        ax_1.plot(data01_axis1, data01_rewards,  color="k", label="score")
        ax_1.set_xlabel("step")
        ax_1.set_ylabel("reward")
        ax_1.legend(loc="upper left")
        plt.savefig(self.save_path+'_reward_graph.png', dpi=300)
        #plt.savefig(self.save_path+'reward_graph.pdf', dpi=300)
        plt.close()

import cv2
import torch
import numpy as np


def min_max(x, mins, maxs, axis=None):
    """_summary_

    Args:
        x (_type_): _description_
        mins (_type_): _description_
        maxs (_type_): _description_
        axis (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    result = (x - mins)/(maxs - mins)
    return result

def make_en_attention(attns, device):
    reshaped_attns = attns[0].view((64, 8, 8)) # rainbow aqt
    reshaped_attns = torch.mean(reshaped_attns, axis=0)
    return reshaped_attns.cpu().detach().numpy()

def make_en_img(attns, raw_img, step, epi_dir, mode="normal"):
    if mode != "mean":
        #raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))
        mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        #mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        #mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_CIVIDIS)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        cv2.imwrite(epi_dir + "/encoder/en_{0:06d}.png".format(step), masked_img)
    else:
        raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))
        mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        #mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_CIVIDIS)
        #mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        cv2.imwrite(epi_dir + "/en_mean.png", masked_img)
    return


def make_de_attention(attns, action_num, device):
    action_attns = []
    for action in range(action_num):
        ac_attn = attns[0, action].view(8, 8) # [0:-1]
        action_attns.append(ac_attn.cpu().detach().numpy())
    return action_attns

def make_de_img(q, attns, step, action_num, epi_dir):#
    raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))

    max_q = q.argmax().item()

    for action in range(action_num):
        if max_q == action:
            txt_color = (0,0,255)
        else:
            txt_color = (0,0,0)

        mask = cv2.resize(attns[action], dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        #mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_CIVIDIS)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)

        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        label_img = np.ones((20,200,3)) * 255
        cv2.imwrite("./results/img/label_img.png", label_img)

        label_img = cv2.imread("./results/img/label_img.png")
        action_name = ac_name_search(action)
        cv2.putText(label_img, text='{0} Q:{1:.3f}'.format(action_name, q[action]), org=(10,15), fontScale=0.5, 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=txt_color, thickness=1, lineType=cv2.LINE_4)# 
        masked_img = cv2.vconcat([masked_img, label_img])

        cv2.imwrite(epi_dir + "/decoder/decoder_act{0}/de{1}-{2:06d}.png".format(action, action, step), masked_img)
    return 


def sk_make_en_attention(args, attns):
    reshaped_attns = attns[0].view((256, 16, 16)) # rainbow aqt
    reshaped_attns = torch.mean(reshaped_attns, axis=0)
    
    #if args.game == "breakout":
    mask_att = torch.ones(16, 16, device=args.device)
    for x in range(16):
        mask_att[0][x] = 0
    return reshaped_attns.cpu().detach().numpy()


def ac_name_search(action):
    
    if action == 0: action_name = "NOOP"
    elif action == 1: action_name = "FRONT"
    elif action == 2: action_name = "LEFT"
    elif action == 3: action_name = "RIGHT"
    
    return action_name



class SUPERVISED_REWARD_LOGGER:
    def __init__(self, path):
        self.save_path = path
        self.train_record_reward = []
        self.train_record_step = []
        self.test_record_reward = []
        self.test_record_step = []
        
        header = ['Step', 'Reward', 'Mean Reward', 'Episode', 'Alpha']
        if not os.path.exists(self.save_path+'train_log.csv'):
            with open(self.save_path + 'train_log.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        
        header = ['Step', 'Mean Reward', 'Episode', 'Alpha']
        if not os.path.exists(self.save_path + 'test_log.csv'):
            with open(self.save_path + 'test_log.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
    
    def record_train_log(self, step, reward, mean_reward, epi, alpha):
        self.train_record_reward.append(reward)
        self.train_record_step.append(step)
        with open(self.save_path+'train_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([step, reward, mean_reward, epi, alpha])
        
    def record_eval_log(self, step, mean_reward, epi, alpha):
        self.test_record_reward.append(mean_reward)
        self.test_record_step.append(step)
        with open(self.save_path+'test_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([step, mean_reward, epi, alpha])

    def _log_plot(self):
        reader = np.loadtxt(self.save_path+'test_log.csv', delimiter=',', skiprows=1, unpack=True)
        step = reader[0]#[row[0] for row in l[1:]]
        reward = reader[1]#[row[1] for row in l[1:]]
        alpha = reader[3]#[row[2] for row in l[1:]]

        data01_axis1 = step
        data01_rewards = reward
        data01_alpha = alpha
        fig_1 = plt.figure(figsize=(12, 6))
        ax_1 = fig_1.add_subplot(111)
        ax_1.plot(data01_axis1, data01_rewards,  color="k", label="score")
        ax_1.plot(data01_axis1, data01_alpha,  color="b", label="alpha")
        ax_1.set_xlabel("step")
        ax_1.set_ylabel("reward")
        ax_1.legend(loc="upper left")
        plt.savefig(self.save_path+'reward_graph.png', dpi=300)