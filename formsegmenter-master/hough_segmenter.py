import torch

import os
from os import path
from os.path import join as pjoin
import numpy as np
import yaml
import sys
import cv2
import pdb

from utils import drawing
from segmenter import Segmenter

green = (0,255,0)
red = (0,0,255)

class HoughSegmenter(Segmenter):
  """docstring for HoughSegmenter."""
  def __init__(self, config, image_folder):
    super(HoughSegmenter, self).__init__(config, image_folder)


  def houghprocess(self, img, predictions, confidence):

    # remove the ones with confidence less than 0.1
    hi_preds = predictions[confidence > 0.1]
    hi_confs = confidence[confidence > 0.1]
    out_img = drawing.draw_sol_np(hi_preds, img.copy(), 0)

    bin_img = cv2.cvtColor(out_img,cv2.COLOR_BGR2GRAY)
    bin_img[:,:] = 0
    bin_img = drawing.draw_sol_np(hi_preds, bin_img, 255)

    ro, theta, threshold = 1, np.pi/180, 40
    """HoughLines outputs a 3dim list of lines where dim1 is the lines,
    dim2 is 1 (pointless) and dim3 is the ro and theta"""
    lines = cv2.HoughLines(bin_img,ro,theta,threshold)
    vertlines = []
    horzlines = []
    for line in lines:
      rho,theta = line[0]
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + out_img.shape[1]*(-b))
      y1 = int(y0 + out_img.shape[0]*(a))
      x2 = int(x0 - out_img.shape[1]*(-b))
      y2 = int(y0 - out_img.shape[0]*(a))
    # minLineLength = 10
    # maxLineGap = 100
    # lines = cv2.HoughLinesP(bin_img,ro,theta,threshold,minLineLength,maxLineGap)
    # print(len(lines))
    # for x1,y1,x2,y2 in lines:
      """only look at the lines that are relatively straight, not diagnol lines.
      look at vertical lines and then horizontal lines"""
      if abs(x1-x2) < 25:
        # print(x1,y1,x2,y2)
        cv2.line(out_img,(x1,y1),(x2,y2),green,2)
        cv2.line(bin_img,(x1,y1),(x2,y2),255,2)
        vertlines.append( ((x1,y1),(x2,y2)) )
      elif abs(y1-y2) < 25:
        # print(x1,y1,x2,y2)
        cv2.line(out_img,(x1,y1),(x2,y2),red,2)
        cv2.line(bin_img,(x1,y1),(x2,y2),255,2)
        horzlines.append( ((x1,y1),(x2,y2)) )

    vertlines = np.asarray(vertlines)
    horzlines = np.asarray(horzlines)

    cv2.imwrite("visuals/{}_predictions.png".format('hough'), out_img)
    cv2.imwrite("visuals/{}_predictions.png".format('bin'), bin_img)

    vertdis = np.asarray(
      [ np.absolute(np.cross(p2-p1,predictions-p1)/np.linalg.norm(p2-p1))
        for p1, p2 in vertlines ]
      )
    horzdis = np.asarray(
      [ np.absolute(np.cross(p2-p1,predictions-p1)/np.linalg.norm(p2-p1))
        for p1, p2 in horzlines ]
      )
    vertdis = 1 / np.amin(vertdis, axis=0)
    horzdis = 1 / np.amin(horzdis, axis=0)
    dis = vertdis + horzdis
    # dis = np.power(dis, 2)
    # avg_dis = np.mean(dis)

    # confidence = confidence/distances
    confidence *= dis
    predictions = predictions[confidence > 0.1]
    confidence = confidence[confidence > 0.1]
    out_img = drawing.draw_sol_np_conf(predictions, confidence, img.copy(), conf_threshold=0.1)
    cv2.imwrite("visuals/{}_predictions.png".format('posthough'), out_img)

    return predictions, confidence


  def get_corners(self, img, predictions, confidence):
    row_projection = predictions[:,1]
    col_projection = predictions[:,0]

    row_clusters = self.proj_nms_single(confidence, row_projection, 10.0)
    col_clusters = self.proj_nms_single(confidence, col_projection, 10.0)

    """sort the clusters by how many points are in each cluster
    with larger clusters are first"""
    row_clusters.sort(key=lambda x:len(x), reverse=True)
    col_clusters.sort(key=lambda x:len(x), reverse=True)

    lengths = [len(x) for x in row_clusters]
    avg = np.mean(lengths)
    mx = np.max(lengths)
    dif = mx - avg
    threshold = avg - dif
    row_clusters = filter(lambda x:len(x)>threshold, row_clusters)

    lengths = [len(x) for x in col_clusters]
    avg = np.mean(lengths)
    mx = np.max(lengths)
    dif = mx - avg
    threshold = avg - dif
    col_clusters = filter(key=lambda x:len(x)>threshold, col_clusters)

    row_clusters = np.array(row_clusters)
    col_clusters = np.array(col_clusters)

    output_grid = np.full((row_clusters.shape[0], col_clusters.shape[0], 2), np.nan)


  def do_image(self, img_path):
    print('doing hough image')
    part_path="visuals/{}_predictions.png".format(img_path.split('/')[-1].split('.')[0])
    big_img, sml_img, s = self.get_img( img_path )
    predictions, confidence = self.get_predictions(sml_img, s, part_path)
    predictions, confidence = self.houghprocess(sml_img, predictions, confidence)
    corners = self.get_corners(sml_img, predictions, confidence)
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

  segmenter = HoughSegmenter(config, image_folder)
  segmenter.segment()

if __name__ == '__main__':
  main()
