
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import continuous_state
from utils import drawing
import numpy as np
import cv2
import json
import yaml
import sys
import os
import math
import csv

from collections import defaultdict

def main():
  with open(sys.argv[1]) as f:
      config = yaml.load(f)

  sol_network_config = config['network']['sol']
  pretrain_config = config['pretraining']
  eval_folder = pretrain_config['validation_set']['img_folder']

  solf = continuous_state.init_model(config)

  if torch.cuda.is_available():
    print("Using GPU")
    solf.cuda()
    dtype = torch.cuda.FloatTensor
  else:
    print("Warning: Not using a GPU, untested")
    dtype = torch.FloatTensor

  writep = config['evaluation']['output_path'].split('_')[0]
  writep = 'data/{}_val'.format(writep)
  if not os.path.exists(writep):
    os.makedirs(writep)

  for fil in os.listdir(eval_folder):
    imgfil = os.path.join(eval_folder, fil)
    org_img = cv2.imread(imgfil, cv2.IMREAD_COLOR)
    if org_img is not None:
      rescale_range = config['pretraining']['sol']['validation_rescale_range']
      target_dim1 = rescale_range[0]

      s = target_dim1 / float(org_img.shape[1])
      target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)
      org_img = cv2.resize(org_img,(target_dim1, target_dim0), interpolation=cv2.INTER_CUBIC)

      img = org_img.transpose([2,1,0])[None,...]
      img = img.astype(np.float32)
      img = torch.from_numpy(img)
      img = img / 128.0 - 1.0

      img = Variable(img.type(dtype), requires_grad=False)

      # print((img))
      predictions = solf(img)

      org_img = img[0].data.cpu().numpy().transpose([2,1,0])
      org_img = ((org_img + 1)*128).astype(np.uint8)
      org_img = org_img.copy()
      org_img = drawing.draw_sol_torch(predictions, org_img)
      predictions = None
      cv2.imwrite(os.path.join(writep, fil), org_img)


if __name__ == "__main__":
  main()
