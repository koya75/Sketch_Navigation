import numpy as np
import torch
import mlagents
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import cv2

class unity_env:
    def __init__(self, file_name, worker_id, time_scale, seed, max_step, device):

        # Unityの進行速度などを設定,大き過ぎるとバグる。
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=time_scale)

        # ゲームの初期化。file_nameでUnityファイルを指定。複数の環境を同時に開く時はworker_idを違うように設定
        self.env = UnityEnvironment(file_name=file_name, seed=seed, worker_id=worker_id, side_channels=[channel])
        self.env.reset()
        
        # これ何なのか忘れた。
        self.behavior_name = list(self.env.behavior_specs)[0]
        spec = self.env.behavior_specs[self.behavior_name].action_spec
        self.action_spase = spec[1][0]

        self.distance = 0
        self.goal_idx = 0
        self.device = torch.device('cuda:{}'.format(device))
        self.max_step = max_step

        self.time_step = 0

    def reset(self):
        # resetで新しいエピソードが開始
        self.env.reset()

        # get_stepsでデータを取得。
        # DecisionStepsとTerminalStepsがあるが、今回Unity内でエピソードの終了条件を設定していないので、TerminalStepsはいらない
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self.time_step = 0

        # decision_steps.obs[0]は画像情報、decision_steps.obs[1]は数値情報。！！！！！！！！！！！！！！！！！！！！！
        # 2つ目の[0]はagentの番号。ML-Agentsは同じ環境をコピペしてagentをいっぱい並んで同時に学習できるが、今回はagent1つしか用意してないので[0]
        obs = self.preproc_img(decision_steps.obs[0][0])
        

        # 数値情報は3つあります
        # 1.残りのサブゴールの数(最終ゴール除く)
        # 2.サブゴールとの角度(0~1)、ゴールが正面にいる時は0.5、左側にいる時は0~0.5、右側にいる時は0.5~1、真後ろにいる時は0か1
        # 3.サブゴールとの距離。角度とスケール統一するため10分の1にしている
        #goal = decision_steps.obs[1][0][1:3]
        self.distance = decision_steps.obs[1][0][-1]
        #self.goal_idx = decision_steps.obs[1][0][0]

        # 最終ゴールに到達しているか、衝突の回数、を記録
        self.info = {'success': False, 'crash_count':0 , 'done':False }
        return obs
    
    def step(self, action):
        # 動作をUnityに入力するためmlagents_envsのActionTuple()にする必要がある。何故かは謎。
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.array([[action]], dtype=np.int32).copy())

        # set_actions()とstep()でagentを動かす
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        # 前と同じ
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        obs = self.preproc_img(decision_steps.obs[0][0])
        #goal = decision_steps.obs[1][0][1:3]
        new_distance = decision_steps.obs[1][0][-1]
        #new_goal_idx = decision_steps.obs[1][0][0]

        # Unity内に設定した報酬を受け取る
        collision = decision_steps[0].reward
        
        # 衝突したらUnity内でrewardが-1にしてます。報酬をUnityで設定するのが面倒くさいのでPythonでやる。
        if collision == -1:
            #reward = -0.3
            if self.info['success'] == False:
                self.info['crash_count']+=1
        """elif reward==1:
            reward = 7
            if self.info['success'] == False:
                self.info['crash_count']+=1
        elif reward == 2:
            reward = 10
        # ゴールに近づいていないと負の報酬を与える。
        if new_distance >= self.distance:
            reward -= 0.1"""

        #dist_reward = torch.sqrt(torch.square(torch.tensor(new_distance, dtype=torch.float32))).to(self.device)
        dist_reward = -new_distance#torch.tensor(new_distance, dtype=torch.float32).to(self.device)
        #dist_reward = 1.0 / (1.0 + dist_reward * dist_reward)
            
        
        """# 残りのサブゴールが1個減った＝サブゴールに到達。
        if self.goal_idx - new_goal_idx == 1. :
            reward = 10.
        # 最終ゴール到達したら、新しい経路とサブゴール(透明の)を生成するようにしてるので、サブゴールの数が増える＝最終ゴール到達。
        # 何故すぐ終了しないのかはエピソードの長さを統一できるため。
        elif self.goal_idx - new_goal_idx <0. :
            reward = 10.
            self.info['success']=True"""
        
        #reward = torch.from_numpy(np.array(reward)).to(self.device).to(torch.float32).unsqueeze(0)

        self.distance = new_distance
        #self.goal_idx = new_goal_idx
        reward = dist_reward#rewards
        self.time_step += 1
        if self.time_step==self.max_step:
            #reward = self.info['crash_count']*-3
            #reward = dist_reward*100
            self.info['done'] = True
        #reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        
        return obs, reward, self.info['done']
    
    def preproc_img(self, img):
        #img = img[:, 24:230]
        #if self.time_step==0:
        #    cv2.imwrite('hoge.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB)*255)
        #img(84, 112)の画像を正方形に圧縮cv2.resize
        #img = resize(img,(128,128),anti_aliasing=True)
        #cv2.imwrite('hoge1.png', img*255)
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img).to(self.device).to(torch.float32)
        return img.unsqueeze(0)
    
    def close(self):
        self.env.close()