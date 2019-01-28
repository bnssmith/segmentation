import torch

import numpy as np
import yaml
import sys
import os
from os import path
from os.path import join as pjoin

from segmenter import Segmenter

def rotationMatrix(degree):
  theta = np.radians(degree)
  c, s = np.cos(theta), np.sin(theta)
  return np.array(((c,-s), (s, c)))

def rotate(mat, degree):
  R = rotationMatrix(degree)
  return np.matmul(mat, R)

def get_xvar(mat, degree, bins):
  rot = rotate(mat, degree)[:,0]
  hist = np.histogram(rot, bins)[0]
  # print(hist)
  hist[hist<3] = 0
  # hist, _, _ = plt.hist(rot, bins)
  # plt.savefig('samplefig{}.png'.format(degree))
  # plt.gcf().clear()
  return hist.var()

def get_yvar(mat, degree, bins):
  rot = rotate(mat, degree)[:,1]
  hist = np.histogram(rot, bins)[0]
  # print(hist)
  hist[hist<3] = 0
  # hist, _, _ = plt.hist(rot, bins)
  # plt.savefig('samplefig{}.png'.format(degree))
  # plt.gcf().clear()
  return hist.var()

def find_best_angle(predictions, imgSize):
  bvsf = 0
  bdsf = 0
  var = 0
  degree_change = 1
  # width / (height/2) * tan(degree_change)
  # the denominator is the approx. width of each bin
  # bins = int( imgSize[1] / (float(imgSize[0]/2) * np.tan(np.radians(degree_change))) )
  # height / (width/2) * tan(degree_change)
  bins = int( imgSize[0] / (float(imgSize[1]/2) * np.tan(np.radians(degree_change))) )
  for deg in range(-10,11,1):
    var = get_xvar(predictions, deg, bins) # collapse on the x axis
    if var > bvsf:
      bvsf = var
      bdsf = deg
    #     print('bvsf: {}, bdsf: {}'.format(bvsf, bdsf))
    # else:
    #     print('var: {}, deg: {}'.format(var, deg))

  degree_change = 0.1
  bins = int( imgSize[0] / (float(imgSize[1]/2) * np.tan(np.radians(degree_change))) )
  bvsf = get_xvar(predictions, bdsf, bins)

  var = bvsf
  bdsfup = bdsf
  varup = get_xvar(predictions, bdsfup+degree_change, bins)
  while varup > var:
    var = varup
    bdsfup += degree_change
    varup = get_xvar(predictions, bdsfup+degree_change, bins)
    # print('var: {}, deg: {}'.format(varup, bdsfup))

  var = bvsf
  bdsfdown = bdsf
  vardown = get_xvar(predictions, bdsfdown-degree_change, bins)
  while vardown > var:
    var = vardown
    bdsfdown -= degree_change
    vardown = get_xvar(predictions, bdsfdown-degree_change, bins)
    # print('var: {}, deg: {}'.format(vardown, bdsfdown))

  if varup > vardown and varup > bvsf:
    bdsf = bdsfup
    bvsf = varup
  elif vardown > varup and vardown > bvsf:
    bdsf = bdsfdown
    bvsf = vardown

  # print('bdsf: {}, bvsf: {}'.format(bdsf, bvsf))
  return rotate(predictions, bdsf), bdsf


class SkewSegmenter(Segmenter):
  """docstring for SkewSegmenter."""
  def __init__(self, config, image_folder):
    super(SkewSegmenter, self).__init__(config, image_folder)


  def postprocess(self, org_img, predictions, confidence):
    predictions, rotation = find_best_angle(predictions, org_img.shape)

    vert_projection = predictions[:,1]

    horz_projection = predictions[:,0]

    vert_clusters = self.proj_nms_single(confidence, vert_projection, 10.0)
    horz_clusters = self.proj_nms_single(confidence, horz_projection, 10.0)

    # sort the clusters by how many points are in each cluster,
    # with larger clusters are first
    vert_clusters.sort(key=lambda x:len(x), reverse=True)
    # take the top vert_cluster_cnt (41) clusters that have the most points
    vert_clusters = np.array(vert_clusters[:self.rows])

    horz_clusters.sort(key=lambda x:len(x), reverse=True)
    horz_clusters = np.array(horz_clusters[:self.cols])

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

    output_grid = np.full((vert_cluster_cnt, horz_cluster_cnt, 2), np.nan)

    red = (0,0,255)
    green = (0,255,0)
    diam = 5
    MATCH_THRESHOLD = 20.0
    threshold = 10.0

    avg = np.mean(np.diff(vert_cluster_medians))
    # print(avg)
    first = vert_cluster_medians[0]
    temp = first
    if draw:
      # c_img = img.copy()
      # c_img = misc.imrotate(c_img, rotation)
      for pt in vert_cluster_medians:
        # cv2.circle(c_img2, (diam, int(pt)), diam, red, 2) # side reds
        cv2.circle(c_img, (diam, int(temp)), diam, red, 2) # side reds
        temp += avg

    for i, c in enumerate(horz_clusters):

      these_preds = predictions[c]
      these_confs = confidence[c]

      horz_median = horz_cluster_medians[i]
      if draw:
        # cv2.circle(c_img2, (int(horz_median), diam), diam, red, 2) # top reds
        cv2.circle(c_img, (int(horz_median), diam), diam, red, 2) # top reds
        # for pt in these_preds:
        #     cv2.circle(c_img2, (int(pt[0]), int(pt[1])), diam, green, 2)

      args = np.argsort(these_preds[:,1])
      these_preds = these_preds[args]
      these_confs = these_confs[args]
      while these_preds[0,1] < first - threshold:
        # output_grid[0,i] = these_preds[0]
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
      if draw:
        for pt in these_preds:
          cv2.circle(c_img, (int(pt[0]), int(pt[1])), diam, green, 2)

    if draw:
      # cv2.imwrite("visuals/{}_3predictions.png".format(part_path), c_img2)
      cv2.imwrite("visuals/{}_2predictions.png".format(part_path), c_img)

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

    # return output_grid
    return rotate(output_grid, -rotation)


def main():
  if len(sys.argv) < 3:
    print(("Usage: ", sys.argv[0], " yaml_config_file image_folder_path"))
    sys.exit()

  with open(sys.argv[1]) as f:
    config = yaml.load(f)

  image_folder = sys.argv[2]

  segmenter = SkewSegmenter(config, image_folder)
  segmenter.segment()


if __name__ == '__main__':
  main()
