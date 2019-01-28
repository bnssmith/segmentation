import numpy as np
import cv2
import math

def draw_sol_torch(predictions, org_img, conf_threshold=0.1):
    # print(predictions.shape)
    for j in range(predictions.size(1)):
        conf = predictions[0,j,0]
        # print(type(conf))
        # print(conf.shape)
        # print(conf.data.cpu().numpy())
        # input('waiting...')
        conf = conf.data.cpu().numpy()
        if conf < conf_threshold:
            continue

        color = int(255*conf)

        pt0 = predictions[0,j,1:3]# * 512

        pt0 = tuple(pt0.data.cpu().numpy().astype(np.int64).tolist())

        x0,y0 = pt0
        mx = int(x0)
        my = int(y0)

        cv2.circle(org_img, (mx, my), 3, color, -1)
    return org_img

def draw_sol_np_conf(predictions, confidence, org_img, conf_threshold=0.1):
    # print(predictions.shape)
    for j in range(predictions.shape[0]):
        conf = confidence[j]
        if conf < conf_threshold:
            continue
        color = int(255*conf)

        x,y = predictions[j]# * 512
        cv2.circle(org_img, (x, y), 3, color, -1)
    return org_img

def draw_sol_np(predictions, org_img, color=0):
    # print(predictions.shape)
    for j in range(predictions.shape[0]):
        x,y = predictions[j]# * 512
        cv2.circle(org_img, (x, y), 3, color, -1)
    return org_img
