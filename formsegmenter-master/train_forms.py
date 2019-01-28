
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
# from PIL import Image
import cv2
import json
import yaml
import sys
import os
import math
import gc

from utils import transformation_utils, drawing


def main():
    dev = '0'
    if len(sys.argv) > 2:
        dev = sys.argv[2]

    with open(sys.argv[1]) as f:
        config = yaml.load(f)

    sol_network_config = config['network']['sol']
    pretrain_config = config['pretraining']
    eval_folder = pretrain_config['validation_set']['img_folder']

    base0 = sol_network_config['base0']
    base1 = sol_network_config['base1']
    solf = StartOfLineFinder(base0, base1)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(dev))
        print("Using GPU")
        solf.cuda()
        dtype = torch.cuda.FloatTensor
    else:
        print("Warning: Not using a GPU, untested")
        dtype = torch.FloatTensor

    alpha_alignment = pretrain_config['sol']['alpha_alignment']
    alpha_backprop = pretrain_config['sol']['alpha_backprop']

    optimizer = torch.optim.Adam(solf.parameters(), lr=pretrain_config['sol']['learning_rate'])

    def train():

        training_set_list = load_file_list(pretrain_config['training_set'])
        train_dataset = SolDataset(training_set_list,
            rescale_range=pretrain_config['sol']['training_rescale_range'],
            transform=CropTransform(pretrain_config['sol']['crop_params']))

        train_dataloader = DataLoader(train_dataset,
            batch_size=pretrain_config['sol']['batch_size'],
            shuffle=True, num_workers=0,
            collate_fn=sol.sol_dataset.collate, pin_memory=True)

        batches_per_epoch = int(pretrain_config['sol']['images_per_epoch']
            /pretrain_config['sol']['batch_size'])
        train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

        if not os.path.exists("snapshots/sol_train"):
            os.makedirs("snapshots/sol_train")

        solf.train()
        sum_loss = 0.0
        steps = 0.0

        for step_i, x in enumerate(train_dataloader):
            img = x['img'].type(dtype)

            sol_gt = None
            if x['sol_gt'] is not None:
                # This is needed because if sol_gt is None it means that there
                # no GT positions in the image. The alignment loss will handle,
                # it correctly as None
                sol_gt = x['sol_gt'].type(dtype)

            predictions = solf(img)
            loss = alignment_loss(predictions, sol_gt, x['label_sizes'],
                alpha_alignment, alpha_backprop)

            org_img = img[0].data.cpu().numpy().transpose([2,1,0])
            org_img = ((org_img + 1)*128).astype(np.uint8)
            org_img = org_img.copy()

            org_img = drawing.draw_sol_torch(predictions, org_img)
            # out = Image.fromarray((org_img * 255).astype(np.uint8))
            # out.save("snapshots/sol_train/{}.png".format(step_i))
            # out = None
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


    def val():

        test_set_list = load_file_list(pretrain_config['validation_set'])
        test_dataset = SolDataset(test_set_list,
            rescale_range=pretrain_config['sol']['validation_rescale_range'],
            transform=None)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=sol.sol_dataset.collate, pin_memory=True)

        writep = config['evaluation']['output_path'].split('_')[0]
        writep = 'snapshots/{}_val'.format(writep)
        if not os.path.exists(writep):
            os.makedirs(writep)

        solf.eval()
        sum_loss = 0.0
        steps = 0.0

        for step_i, x in enumerate(test_dataloader):
            img = x['img'].type(dtype)
            if x['sol_gt'] is not None:
                # This is needed because if sol_gt is None it means that there
                # no GT positions in the image. The alignment loss will handle,
                # it correctly as None
                sol_gt = x['sol_gt'].type(dtype)

            predictions = solf(img)
            loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)

            # Write images to file to visualization
            org_img = img[0].data.cpu().numpy().transpose([2,1,0])
            org_img = ((org_img + 1)*128).astype(np.uint8)
            org_img = org_img.copy()
            org_img = drawing.draw_sol_torch(predictions, org_img)
            # out = Image.fromarray((org_img * 255).astype(np.uint8))
            # out.save(os.path.join(writep, "{}.png".format(step_i)))
            cv2.imwrite(os.path.join(writep, "{}.png".format(step_i)), org_img)

            sum_loss += loss.data.cpu().numpy()
            steps += 1
            predictions = None
            loss = None

            gc.collect()

        return sum_loss/steps


    if not os.path.exists(pretrain_config['snapshot_path']):
        os.makedirs(pretrain_config['snapshot_path'])

    lowest_loss = np.inf
    for epoch in range(20):
        print(("Epoch", epoch))

        loss = train()
        print(("Train Loss", loss))

        loss = val()
        print(("Validation Loss", loss))

        if lowest_loss >= loss:
            lowest_loss = loss
            print ("Saving Best")
            torch.save(solf.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'sol.pt'))

if __name__ == '__main__':
    main()
