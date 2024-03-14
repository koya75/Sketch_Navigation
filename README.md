# Sketch_Navigation

スケッチ指示に追従させるように強化学習を用いて動作を決定する．

# ディレクトリ
```
Sketch_Navigation/
├── docker      ：Dockerに関するファイルが格納されている．
├── dqn         ：DQNを用いたコードが格納されている．（メインフォルダ）
│   ├── main    ：強化学習に関するコードが格納されている．
│   ├── models  ：モデルに関するコードが格納されている．
│   ├── results ：実験結果が保存されるフォルダ．
│   └── sketch  ：スケッチ指示を格納するフォルダ．
├── envs        ：環境用のコードが格納されている．
│   └── apps    ：Unityを用いてビルドしたファイルを格納．
└── ml-agents   ：Unityを用いた機械学習用のツール．
```

# 実行方法
## Dockerの起動
Sketch_Navigation/で以下を実行し，docker imageを作成します．
```
bash docker/build.sh
```
作成したら以下を実行し，dockerに入ります．
```
bash docker/run.sh
```
このときrun.shの`--mount type=bind,source="/data1/results",target=/home/${USERNAME}/Mask-Attention_by_PPO/results`部分でマウントを行うので，`source="/data1/results"`を自分のフォルダに変更してください．

## プログラムの実行方法
docker上（dqn/）で以下を実行します．
```
./train.sh
```
これで実験自体は動かすことができます．

# 引数の設定
run_target_franka_ppo_model_mask_double_lstm.shの中に実験に使用する引数が記載されているので都度変更してください．
```
model=target_mask_double                    どのモデルを使うか
robot=128                                   何台のロボットで実験するか

--outdir results/franka/${model}_lstm       保存先のフォルダ
--model ${model}                            モデルの指定
--epochs 10                                 エポック数
--gamma 0.99                                割引率
--step_offset 0                             ステップオフセット（途中から学習する場合どのステップから始めるか指定）
--lambd 0.995                               ラムダの値
--lr 0.0002                                 学習率
--max-grad-norm 40                          最大勾配
--gpu 2                                     何番のGPUを使うか
--use-lstm                                  LSTMを使うかどうか（）
--num-envs ${robot}                         ロボットの数
--eval-n-runs ${robot}                      評価時のロボットの数
--update-batch-interval 1                   更新の頻度
--num-items 3                               アイテムの数
--item-names item21 item25 item38           どのアイテム使うか
--target item21                             把持対象物体
--isaacgym-assets-dir /opt/isaacgym/assets  アセットの指定
--item-urdf-dir ./urdf                      urdfファイルの指定
--steps 25000000                            maxステップ
--eval-batch-interval 20                    評価のタイミング指定
--descentstep 12                            ハンドが何ステップで下がるか
--mode normal                               ノーマルかハードか
--hand                                      ハンドカメラを使う（無いと固定カメラ）
```