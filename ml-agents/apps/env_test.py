import numpy as np
import mlagents
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from skimage.transform import resize

# 基本 https://github.com/Unity-Technologies/ml-agents/blob/2.0-verified/docs/Python-API.md を参照

class unity_env:
    def __init__(self, file_name, worker_id, time_scale, seed):

        # Unityの進行速度などを設定,大き過ぎるとバグる。
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=time_scale)

        # ゲームの初期化。file_nameでUnityファイルを指定。複数の環境を同時に開く時はworker_idを違うように設定
        self.env = UnityEnvironment(file_name=file_name, seed=seed, worker_id=worker_id, side_channels=[channel])
        self.env.reset()
        
        # これ何なのか忘れた。
        self.behavior_name = list(self.env.behavior_specs)[0]

        self.distance = 0
        self.goal_idx = 0

    def reset(self):
        # resetで新しいエピソードが開始
        self.env.reset()

        # get_stepsでデータを取得。
        # DecisionStepsとTerminalStepsがあるが、今回Unity内でエピソードの終了条件を設定していないので、TerminalStepsはいらない
        decision_steps, _ = self.env.get_steps(self.behavior_name)

        # decision_steps.obs[0]は画像情報、decision_steps.obs[1]は数値情報。！！！！！！！！！！！！！！！！！！！！！
        # 2つ目の[0]はagentの番号。ML-Agentsは同じ環境をコピペしてagentをいっぱい並んで同時に学習できるが、今回はagent1つしか用意してないので[0]
        obs = self.preproc_img(decision_steps.obs[0][0])

        # 数値情報は3つあります
        # 1.残りのサブゴールの数(最終ゴール除く)
        # 2.サブゴールとの角度(0~1)、ゴールが正面にいる時は0.5、左側にいる時は0~0.5、右側にいる時は0.5~1、真後ろにいる時は0か1
        # 3.サブゴールとの距離。角度とスケール統一するため10分の1にしている
        goal = decision_steps.obs[1][0][1:3]
        self.distance = decision_steps.obs[1][0][-1]
        self.goal_idx = decision_steps.obs[1][0][0]

        # 最終ゴールに到達しているか、衝突の回数、を記録
        self.info = {'success': False, 'crash_count':0 }
        return obs, goal
    
    def step(self, action):
        # 動作をUnityに入力するためmlagents_envsのActionTuple()にする必要がある。何故かは謎。
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.array([[action]], dtype=np.int32))

        # set_actions()とstep()でagentを動かす
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        # 前と同じ
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        obs = self.preproc_img(decision_steps.obs[0][0])
        goal = decision_steps.obs[1][0][1:3]
        new_distance = decision_steps.obs[1][0][-1]
        new_goal_idx = decision_steps.obs[1][0][0]

        # Unity内に設定した報酬を受け取る
        reward = decision_steps[0].reward
        
        # 衝突したらUnity内でrewardが-1にしてます。報酬をUnityで設定するのが面倒くさいのでPythonでやる。
        if reward == -1:
            reward = -3.
            if self.info['success'] == False:
                self.info['crash_count']+=1
        # ゴールに近づいていないと負の報酬を与える。
        if new_distance >= self.distance:
            reward += -0.01
        
        # 残りのサブゴールが1個減った＝サブゴールに到達。
        if self.goal_idx - new_goal_idx == 1. :
            reward = 10.
        # 最終ゴール到達したら、新しい経路とサブゴール(透明の)を生成するようにしてるので、サブゴールの数が増える＝最終ゴール到達。
        # 何故すぐ終了しないのかはエピソードの長さを統一できるため。
        elif self.goal_idx - new_goal_idx <0. :
            reward = 10.
            self.info['success']=True

        self.distance = new_distance
        self.goal_idx = new_goal_idx
        return obs, goal, reward, self.info
    
    def preproc_img(self, img):
        #img(84, 112)の画像を正方形に圧縮
        img = resize(img,(84,84),anti_aliasing=True)
        img = img.transpose((2,0,1))
        return img
    
    def close(self):
        self.env.close()

test_env = unity_env(file_name='Indoor', worker_id=0, time_scale=1, seed=123)
n_test = 10
ep_len = 100
for _ in range(n_test):
    obs, state = test_env.reset()
    for i in range(ep_len):
        action = np.random.randint(4,dtype=np.int32)
        next_obs, next_state, reward, info = test_env.step(action)
test_env.close()