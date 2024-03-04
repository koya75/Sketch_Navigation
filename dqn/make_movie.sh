#!/bin/bash
model=SKT
folder=17
env=known
#master_graduation
#eval

#for folder in {0..9}
#do
for epi in {1..10}
do
    python3 make_movie.py \
        --mode raw \
        --load-dir /data1/honda/results/DQN/master_graduation/${env}/${model}/${folder}/epi${epi}/

    python3 make_movie.py \
        --mode raw_point \
        --load-dir /data1/honda/results/DQN/master_graduation/${env}/${model}/${folder}/epi${epi}/

    python3 make_movie.py \
        --mode encoder \
        --load-dir /data1/honda/results/DQN/master_graduation/${env}/${model}/${folder}/epi${epi}/

    python3 make_movie.py \
        --mode decoder \
        --load-dir /data1/honda/results/DQN/master_graduation/${env}/${model}/${folder}/epi${epi}/
done
#done