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
from os import path
from os.path import join as pjoin

from deskew import *

from collections import defaultdict


class Segmenter(object):
  """docstring for Segmenter."""
  def __init__(self, config, image_folder):
    super(Segmenter, self).__init__()
    self.config = config
    self.image_folder = image_folder
    self.sol_network_config = config['network']['sol']
    self.pretrain_config = config['pretraining']
    self.outpath = config['evaluation']['output_path']

    self.rows = int(config['evaluation']['rows'])
    self.columns = int(config['evaluation']['columns'])

    self.network = continuous_state.init_model(config)


  def proj_nms_single(self, confidence, x_points, overlap_thresh):
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


  def postprocess(self, org_img, predictions, confidence, part_path=''):

    vert_projection = predictions[:,1]

    horz_projection = predictions[:,0]

    vert_clusters = self.proj_nms_single(confidence, vert_projection, 10.0)
    horz_clusters = self.proj_nms_single(confidence, horz_projection, 10.0)

    # sort the clusters by how many points are in each cluster,
    # with larger clusters are first
    vert_clusters.sort(key=lambda x:len(x), reverse=True)
    # take the top self.rows (41) clusters that have the most points
    vert_clusters = np.array(vert_clusters[:self.rows])

    horz_clusters.sort(key=lambda x:len(x), reverse=True)
    horz_clusters = np.array(horz_clusters[:self.columns])

    # medians as center of each cluster
    vert_cluster_medians = np.array([np.median(vert_projection[c])
      for c in vert_clusters])
    horz_cluster_medians = np.array([np.median(horz_projection[c])
      for c in horz_clusters])

    # sort the points
    vert_mean_sort_idx = vert_cluster_medians.argsort()
    horz_mean_sort_idx = horz_cluster_medians.argsort()

    vert_cluster_medians = vert_cluster_medians[vert_mean_sort_idx]
    horz_cluster_medians = horz_cluster_medians[horz_mean_sort_idx]

    # sorts the clusters, according to the median points
    vert_clusters = vert_clusters[vert_mean_sort_idx]
    horz_clusters = horz_clusters[horz_mean_sort_idx]

    output_grid = np.full((self.rows, self.columns, 2), np.nan)

    red = (0,0,255)
    green = (0,255,0)
    diam = 5
    MATCH_THRESHOLD = 20.0

    avg = np.mean(np.diff(vert_cluster_medians))
    threshold = avg / 5
    first = vert_cluster_medians[0]

    for i, c in enumerate(horz_clusters):

      these_preds = predictions[c]
      these_confs = confidence[c]

      horz_median = horz_cluster_medians[i]

      args = np.argsort(these_preds[:,1])
      these_preds = these_preds[args]
      these_confs = these_confs[args]
      while these_preds[0,1] < first - threshold:
        these_preds = these_preds[1:]
        these_confs = these_confs[1:]
      output_grid[0,i] = these_preds[0]
      if these_preds[0,1] > first + threshold:
        output_grid[0,i] = (horz_median, first)

      dists = np.concatenate(
        (np.diff(these_preds[:,1], axis=0) < avg-threshold, [False]))
      rems = []
      for j, dis in enumerate(dists):
        if dis:
          if these_confs[j] > these_confs[j+1]:
            rems += [j+1]
          else:
            rems += [j]
      these_preds = np.delete(these_preds, rems, axis=0)

      k = 1
      for j, pt in enumerate(these_preds[1:]):
        if k >= output_grid.shape[0]:
          break
        if pt[1] < output_grid[k-1,i][1] + avg-threshold:
          continue
        elif pt[1] > output_grid[k-1,i][1] + avg+threshold:
          k += 1
        else:
          output_grid[k,i] = pt
          k += 1

    nan_idxs = np.where(np.isnan(output_grid[:,:,0]))
    for i,j in zip(*nan_idxs):

        i_neg = i-1
        while i_neg > -1 and np.isnan(output_grid[i_neg, j, 0]):
            i_neg -= 1
        i_neg = None if i_neg == -1 else i_neg

        i_pos = i+1
        while i_pos < output_grid.shape[0] and np.isnan(output_grid[i_pos, j, 0]):
            i_pos += 1
        i_pos = None if i_pos == output_grid.shape[0] else i_pos


        j_neg = j-1
        while j_neg > -1 and np.isnan(output_grid[i, j_neg, 0]):
            j_neg -= 1
        j_neg = None if j_neg == -1 else j_neg

        j_pos = j+1
        while j_pos < output_grid.shape[1] and np.isnan(output_grid[i, j_pos, 0]):
            j_pos += 1
        j_pos = None if j_pos == output_grid.shape[1] else j_pos


        assert i_pos is not None or i_neg is not None
        assert j_pos is not None or j_neg is not None
        if i_pos is None:
            i_pos = i_neg

        if i_neg is None:
            i_neg = i_pos

        if j_pos is None:
            j_pos = j_neg

        if j_neg is None:
            j_neg = j_pos

        # This doees the averaging part described above
        output_grid[i,j,0] = (output_grid[i_pos, j,0] +
            output_grid[i_neg, j,0])/2.0
        output_grid[i,j,1] = (output_grid[i, j_pos,1] +
            output_grid[i, j_neg,1])/2.0

    return output_grid


  def get_img(self, image_path):
    org_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rescale_range = self.config['pretraining']['sol']['validation_rescale_range']
    target_dim1 = rescale_range[0]

    s = target_dim1 / float(org_img.shape[1])
    target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)
    full_res_img = org_img
    org_img = cv2.resize(org_img,(target_dim1, target_dim0),
      interpolation = cv2.INTER_CUBIC)
    return full_res_img, org_img, s


  def get_predictions(self, org_img, s, write_path=''):
    img = org_img.transpose([2,1,0])[None,...]
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = img / 128.0 - 1.0

    if torch.cuda.is_available():
      img = img.cuda()

    predictions = self.network(img)
    if write_path is not '':
      org_img = img[0].data.cpu().numpy().transpose([2,1,0])
      org_img = ((org_img + 1)*128).astype(np.uint8)
      c_img = org_img.copy()
      c_img = drawing.draw_sol_torch(predictions, c_img)
      cv2.imwrite(write_path, c_img)

    predictions = predictions.squeeze(0).data.cpu().numpy()
    # print(predictions.shape)

    # predictions is a matrix of 2000ish x 5
    # where the first is the confidence and then there are two pairs of x and y coordinates
    # here we are extracting just the first pair since the second is probably a repeat
    confidence = predictions[:,0]
    predictions = predictions[:,1:3]
    return predictions, confidence


  def write_points(self, corners, c_img, s, img_path, draw=False):

    if draw:
      if not os.path.exists("visuals"):
        os.makedirs("visuals")

    part_path = img_path.split('/')[-1].split('.')[0]
    with open(self.outpath, 'a') as f:
      writer = csv.writer(f, delimiter='\t', quotechar='|')

      if draw:
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

          if draw:
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
    if draw:
      cv2.imwrite("visuals/{}.png".format(part_path), c_img)

  def write_headers(self):
    with open(self.outpath, 'w') as f:
      f.write('')

  def segment(self):
    self.write_headers()

    if os.path.isdir(self.image_folder):

      for f in os.listdir(self.image_folder):
        if os.path.isdir(pjoin(self.image_folder, f)):

          for f2 in os.listdir(pjoin(self.image_folder, f)):
            if ".jpg" in f2:
              do_image(pjoin(self.image_folder, f, f2))

        else:
          self.do_image(pjoin(self.image_folder, f))

    else:
      self.do_image(pjoin(self.image_folder))


  def do_image(self, img_path):
    print('doing image')
    big_img, sml_img, s = self.get_img( img_path )
    predictions, confidence = self.get_predictions(sml_img, s, write_path='visuals/show_preds.png')
    # remove the ones with confidence less than 0.1
    predictions = predictions[confidence > 0.1]
    confidence = confidence[confidence > 0.1]
    corners = self.postprocess(sml_img, predictions, confidence)
    self.write_points(corners, big_img, s, img_path, True)
    print('wrote image')

def main():
  if len(sys.argv) < 3:
    print(("Usage: ", sys.argv[0], " yaml_config_file image_folder_path"))
    sys.exit()

  with open(sys.argv[1]) as f:
    config = yaml.load(f)

  image_folder = sys.argv[2]

  segmenter = Segmenter(config, image_folder)
  segmenter.segment()


if __name__ == '__main__':
  main()
