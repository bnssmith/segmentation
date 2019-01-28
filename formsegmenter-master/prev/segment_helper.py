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

from deskew import *

from collections import defaultdict

def proj_nms_single(confidence, x_points, overlap_thresh):
# Based on
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Maybe could port to pytorch to work over the tensors directly

    idxs = np.argsort(confidence)
    clusters = []

    pick = []
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]

        # xx is all of the x points that have a lower cnfidence than the
        # current x we are looking at
        xx = x_points[idxs[:last+1]]
        this_x = x_points[i]

        # the list of distances between the current x and all of the
        # other x's with lower confidence`
        dis = np.abs(xx - this_x)
        # gets all of the xx points within a certain distance of the current x
        matches = np.where(dis < overlap_thresh)[0]
        matched_idxs =  idxs[matches]
        # takes all of the 'close' points and clusters them into a
        # list and adds that list to the clusters list
        clusters.append(matched_idxs)
        idxs = np.delete(idxs, matches)

    # returns a list of lists of points, so that each cluster in clusters
    # conatains a bunch of points that are close to each other
    return clusters


def get_img(image_path, config):
    org_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rescale_range = config['pretraining']['sol']['validation_rescale_range']
    target_dim1 = rescale_range[0]

    s = target_dim1 / float(org_img.shape[1])
    target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)
    full_res_img = org_img
    org_img = cv2.resize(org_img,(target_dim1, target_dim0),
        interpolation = cv2.INTER_CUBIC)
    return full_res_img, org_img, s


def get_corners(org_img, sol, s, part_path='', draw=False):
    img = org_img.transpose([2,1,0])[None,...]
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = img / 128.0 - 1.0

    img = Variable(img, requires_grad=False, volatile=True).cuda()

    predictions = sol(img)
    if draw:
        org_img = img[0].data.cpu().numpy().transpose([2,1,0])
        org_img = ((org_img + 1)*128).astype(np.uint8)
        c_img = org_img.copy()
        c_img = drawing.draw_sol_torch(predictions, c_img)
        cv2.imwrite("visuals/{}_predictions.png".format(part_path), c_img)

    predictions = predictions.data.cpu().numpy()
    # remove the ones with confidence less than 0.1
    predictions = predictions[predictions[:,:,0] > 0.1]

    # predictions is a matrix of 2000ish x 5
    # where the first is the confidence and then there are two pairs of x and y coordinates
    # here we are extracting just the first pair since the second is probably a repeat
    confidence = predictions[:,0]
    predictions = predictions[:,1:3]
    return predictions, confidence


def write_points(corners, c_img, part_path, s, outpath, draw=False, draw2=False):

    if draw2:
        if not os.path.exists("visuals"):
            os.makedirs("visuals")

    # part_path = img_path.split('/')[-1].split('.')[0]
    with open(outpath, 'a') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='|')

        if draw2:
            red = (0,0,255)
            diam = 10
        for i in range(corners.shape[0]-1):
            row = [part_path]
            for j in range(corners.shape[1]-1):
                pt0 = corners[i,j] / s
                pt1 = corners[i+1,j] / s
                pt2 = corners[i,j+1] / s
                pt3 = corners[i+1,j+1] / s

                min_x = int(min([pt0[0], pt1[0], pt2[0], pt3[0]]))
                max_x = int(max([pt0[0], pt1[0], pt2[0], pt3[0]]))

                min_y = int(min([pt0[1], pt1[1], pt2[1], pt3[1]]))
                max_y = int(max([pt0[1], pt1[1], pt2[1], pt3[1]]))

                row += [min_x, max_x, min_y, max_y]

                if draw2:
                    cv2.circle(c_img, (min_x, min_y), diam, red, 2)
                    if i == corners.shape[0]-2 and j == corners.shape[1]-2:
                        cv2.circle(c_img, (max_x, max_y), diam, red, 2)
                        cv2.circle(c_img, (min_x, max_y), diam, red, 2)
                        cv2.circle(c_img, (max_x, min_y), diam, red, 2)
                    elif i == corners.shape[0]-2:
                        cv2.circle(c_img, (min_x, max_y), diam, red, 2)
                    elif j == corners.shape[1]-2:
                        cv2.circle(c_img, (max_x, min_y), diam, red, 2)

            # raw_input(row)
            writer.writerow(row)

    if draw2:
        cv2.imwrite("visuals/{}.png".format(part_path), c_img)

def write_headers(outpath):
    with open(outpath, 'w') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='|')
        writer.writerow(['image',
        'house_number', 'xmax', 'ymin', 'ymax',
        'visit_number', 'xmax', 'ymin', 'ymax',
        'family_number', 'xmax', 'ymin', 'ymax',
        'name', 'xmax', 'ymin', 'ymax',
        'relation', 'xmax', 'ymin', 'ymax',
        'ownership', 'xmax', 'ymin', 'ymax',
        'home_value', 'xmax', 'ymin', 'ymax',
        'radio', 'xmax', 'ymin', 'ymax',
        'farm', 'xmax', 'ymin', 'ymax',
        'sex', 'xmax', 'ymin', 'ymax',
        'race', 'xmax', 'ymin', 'ymax',
        'age', 'xmax', 'ymin', 'ymax',
        'maritial_status', 'xmax', 'ymin', 'ymax',
        'marriage_age', 'xmax', 'ymin', 'ymax',
        'education', 'xmax', 'ymin', 'ymax',
        'read/write', 'xmax', 'ymin', 'ymax',
        'birthplace', 'xmax', 'ymin', 'ymax',
        'fthr_birthplace', 'xmax', 'ymin', 'ymax',
        'mthr_birthplace', 'xmax', 'ymin', 'ymax',
        'first_language', 'xmax', 'ymin', 'ymax',
        'code1', 'xmax', 'ymin', 'ymax',
        'code2', 'xmax', 'ymin', 'ymax',
        'code3', 'xmax', 'ymin', 'ymax',
        'immigration_yr', 'xmax', 'ymin', 'ymax',
        'naturalization', 'xmax', 'ymin', 'ymax',
        'english', 'xmax', 'ymin', 'ymax',
        'occupation', 'xmax', 'ymin', 'ymax',
        'industry', 'xmax', 'ymin', 'ymax',
        'code4', 'xmax', 'ymin', 'ymax',
        'class', 'xmax', 'ymin', 'ymax',
        'employed', 'xmax', 'ymin', 'ymax',
        'line_num', 'xmax', 'ymin', 'ymax',
        'veteran', 'xmax', 'ymin', 'ymax',
        'war', 'xmax', 'ymin', 'ymax',
        'farm_schedule', 'xmax', 'ymin', 'ymax'])

#
#
#
#
#
#
#
#
