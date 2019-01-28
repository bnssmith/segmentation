
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sol
from sol.start_of_line_finder import StartOfLineFinder
from sol.alignment_loss import alignment_loss
from sol.sol_dataset import SolDataset
from sol.crop_transform import CropTransform

from utils.dataset_wrapper import DatasetWrapper
from utils.dataset_parse import load_file_list

import numpy as np
import cv2
import json
import yaml
import sys
import os
import math
import gc

from utils import transformation_utils, drawing

with open(sys.argv[1]) as f:
    config = yaml.load(f)

sol_network_config = config['network']['sol']
pretrain_config = config['pretraining']
eval_folder = pretrain_config['validation_set']['img_folder']

base0 = sol_network_config['base0']
base1 = sol_network_config['base1']
solf = StartOfLineFinder(base0, base1)
if torch.cuda.is_available():
    print("Using GPU")
    solf.cuda()
    dtype = torch.cuda.FloatTensor
else:
    print("Warning: Not using a GPU, untested")
    dtype = torch.FloatTensor

alpha_alignment = pretrain_config['sol']['alpha_alignment']
alpha_backprop = pretrain_config['sol']['alpha_backprop']

optimizer = torch.optim.Adam(solf.parameters(), lr=pretrain_config['sol']['learning_rate'])

def train(config):

    training_set_list = load_file_list(pretrain_config['training_set'])
    train_dataset = SolDataset(training_set_list,
        rescale_range=pretrain_config['sol']['training_rescale_range'],
        transform=CropTransform(pretrain_config['sol']['crop_params']))

    train_dataloader = DataLoader(train_dataset,
        batch_size=pretrain_config['sol']['batch_size'],
        shuffle=True, num_workers=0,
        collate_fn=sol.sol_dataset.collate)

    batches_per_epoch = int(pretrain_config['sol']['images_per_epoch']
        /pretrain_config['sol']['batch_size'])
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

    if not os.path.exists("snapshots/sol_train"):
        os.makedirs("snapshots/sol_train")

    solf.train()
    sum_loss = 0.0
    steps = 0.0

    for step_i, x in enumerate(train_dataloader):
        img = Variable(x['img'].type(dtype), requires_grad=False)

        sol_gt = None
        if x['sol_gt'] is not None:
            # This is needed because if sol_gt is None it means that there
            # no GT positions in the image. The alignment loss will handle,
            # it correctly as None
            sol_gt = Variable(x['sol_gt'].type(dtype), requires_grad=False)

        # print((img.shape))
        predictions = solf(img)
        loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)

        org_img = img[0].data.cpu().numpy().transpose([2,1,0])
        org_img = ((org_img + 1)*128).astype(np.uint8)
        org_img = org_img.copy()

        org_img = drawing.draw_sol_torch(predictions, org_img)
        cv2.imwrite("snapshots/sol_train/{}.png".format(step_i), org_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.data.cpu().numpy()
        steps += 1
        predictions = None
        loss = None

        gc.collect()

    return sum_loss/steps


def val(config):
    writep = config['evaluation']['output_path'].split('_')[0]
    writep = 'snapshots/{}_val'.format(writep)
    if not os.path.exists(writep):
        os.makedirs(writep)

    with open(pretrain_config['validation_set']['file_list'], 'r') as f:
        evals = json.load(f)

    solf.eval()

    for fil in evals:
        imgfil = os.path.join(eval_folder, fil[1])
        org_img = cv2.imread(imgfil, cv2.IMREAD_COLOR)
        if org_img is not None:
            rescale_range = pretrain_config['sol']['validation_rescale_range']
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
            cv2.imwrite(os.path.join(writep, fil[1]), org_img)
            predictions = None

if not os.path.exists(pretrain_config['snapshot_path']):
    os.makedirs(pretrain_config['snapshot_path'])

lowest_loss = np.inf
for epoch in range(10):
    print(("Epoch", epoch))

    loss = train(config)
    print(("Train Loss", loss))

    val(config)
    print("Validated")

    torch.save(solf.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'sol.pt'))
