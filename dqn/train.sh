#!/bin/bash
model=SKT
env=ManyManyBoxNav #Indoor_bird, ManyManyBoxNav

# Start tuning hyperparameters
python3 train.py \
    --seed 123 \
    --model ${model} \
    --n-epochs 1000000 \
    --n-cycles 50 \
    --n-test 10 \
    --action-size 4 \
    --sketch-size 1 \
    --buffer-size 250000 \
    --batch-size 32 \
    --replay-frequency 4 \
    --device 0 \
    --env-name envs/apps/${env}\
    --save-direct results/DQN/ \
    --time-scale 20 \
    --super-model 5 \