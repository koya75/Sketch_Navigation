import os
from tqdm import tqdm
import cv2
import numpy as np
import csv
import argparse

import warnings
warnings.simplefilter('ignore')

# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow AQT')
parser.add_argument('--mode', type=str, default='raw', choices=['raw', 'encoder', 'decoder', 'raw_point', 'sketch_encoder', 'sketch_action_decoder'], metavar='CUDA', help='Cuda Device')
parser.add_argument('--load-dir', type=str, default='visuals/breakout_rainbow_aqt/epi1/', help='Load data')

# Setup
args = parser.parse_args()

# Print options
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

model="SKT" # Vanilla SKT
folder=18
env="known" #known unknown

print("Make movie: {}".format(args.mode))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

if model == "SKT":
  load_dir = '/data1/honda/results/DQN/master_graduation/{}/{}/{}/epi{}/'.format(env, model, folder, 2)
  movie_path = os.path.join(load_dir, "select_decoder_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 10.0, (200, 220))

  file_num = sum(os.path.isfile(os.path.join(load_dir + "decoder_/", name)) for name in os.listdir(load_dir + "decoder_/"))
  for idx in tqdm(range(file_num)):
    en_att = cv2.imread(load_dir + "decoder_/de-{0:06d}.png".format(idx))
    video.write(en_att)
  video.release()

#if args.mode == "raw":
for epi in range(10):
  load_dir = '/data1/honda/results/DQN/master_graduation/{}/{}/{}/epi{}/'.format(env, model, folder, epi+1)
  movie_path = os.path.join(load_dir, "raw_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 10.0, (128, 128))

  file_num = sum(os.path.isfile(os.path.join(load_dir + "raw_img/", name)) for name in os.listdir(load_dir + "raw_img/"))
  for idx in tqdm(range(file_num)):
    raw_img = cv2.imread(load_dir + "raw_img/raw_{0:06d}.png".format(idx))
    video.write(raw_img)
  video.release()

#elif args.mode == "raw_point":
  movie_path = os.path.join(load_dir, "raw_point_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 10.0, (128, 128))

  file_num = sum(os.path.isfile(os.path.join(load_dir + "raw_img_point/", name)) for name in os.listdir(load_dir + "raw_img_point/"))
  for idx in tqdm(range(file_num)):
    raw_img = cv2.imread(load_dir + "raw_img_point/raw_{0:06d}.png".format(idx))
    video.write(raw_img)
  video.release()

#elif args.mode == "encoder":
  if model == "SKT":
    movie_path = os.path.join(load_dir, "encoder_movie.mp4")
    video = cv2.VideoWriter(movie_path, fourcc, 10.0, (200, 200))

    file_num = sum(os.path.isfile(os.path.join(load_dir + "encoder/", name)) for name in os.listdir(load_dir + "encoder/"))
    for idx in tqdm(range(file_num)):
      en_att = cv2.imread(load_dir + "encoder/en_{0:06d}.png".format(idx))
      video.write(en_att)
    video.release()

  #elif args.mode == "decoder":
    label_img = np.ones((220,5,3)) * 125
    cv2.imwrite("/data1/honda/results/img/mv_img.png", label_img)

    if not os.path.exists(os.path.join(load_dir, "movie_act_img")):
      os.makedirs(os.path.join(load_dir, "movie_act_img"))
    act_n = 4

    movie_path = os.path.join(load_dir, "decoder_movie.mp4")
    video = cv2.VideoWriter(movie_path, fourcc, 10.0, (200*act_n + 5*(act_n-1), 220))

    file_num = sum(os.path.isfile(os.path.join(load_dir + "decoder/decoder_act0/", name)) for name in os.listdir(load_dir + "decoder/decoder_act0/"))
    mv_img = cv2.imread("/data1/honda/results/img/mv_img.png")

    for idx in tqdm(range(file_num)):
      att_img = cv2.imread(load_dir + "decoder/decoder_act0/de0-{0:06d}.png".format(idx))
      for act in range(1, act_n):
        att_img = cv2.hconcat([att_img, mv_img])
        act_img = cv2.imread(load_dir + "decoder/decoder_act{0}/de{1}-{2:06d}.png".format(act,act,idx))
        att_img = cv2.hconcat([att_img, act_img])
      cv2.imwrite(load_dir + "movie_act_img/act-{0:06d}.png".format(idx), att_img)
      #print(att_img.shape)
      video.write(att_img)
    video.release()

  """for act in range(act_n):
    movie_path = os.path.join(load_dir, "decoder_movie/decoder_movie{0}.mp4".format(act))
    video = cv2.VideoWriter(movie_path, fourcc, 2.0, (200, 200))

    file_num = sum(os.path.isfile(os.path.join(load_dir + "decoder/decoder_act{0}/".format(act), name)) for name in os.listdir(load_dir + "decoder/decoder_act{0}/".format(act)))
    for idx in tqdm(range(file_num)):
      de_att = cv2.imread(load_dir + "decoder/decoder_act{0}/de{1}_{2:06d}.png".format(act,act,idx))
      video.write(de_att)
    video.release()"""