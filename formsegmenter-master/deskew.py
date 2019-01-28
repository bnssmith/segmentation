import sys
import numpy as np
# import matplotlib.pyplot as plt
import time
from scipy import misc
from scipy import ndimage
import os
import cv2

def deskew(img):
    org_img = img
    start_time = time.time()

    start_angle_range = 4.0
    offset = start_angle_range * 2.0
    best_angle = offset

    img = rotate(img, -best_angle)

    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)

    pyr = [img]
    while list(range(3)):
        tmp_img = cv2.pyrDown(pyr[-1])
        if max(tmp_img.shape[:2]) < 500:
            break
        pyr.append(tmp_img)

    start = 0
    end_goal = 10
    end_denom = 2-(2**(-(len(pyr)-1)))
    for j, img in enumerate(reversed(pyr)):
        result_cache = {}
        end = int(np.ceil(((2-2**(-j))/end_denom)*end_goal))
        for i in range(start, end):
            angle_range = start_angle_range * (0.5)**i
            best_angle = find_best_angle(img, best_angle, angle_range, angle_range/2, result_cache = result_cache)
        start = end-1
    # print "Angle: ", best_angle - offset
    # print "Time:  ", time.time() - start_time

    return rotate(org_img, best_angle+-offset), (best_angle - offset)

def deskew_simple(img):
    org_img = img
    start_time = time.time()

    start_angle_range = 4.0
    offset = start_angle_range * 2.0


    img = rotate(img, -offset)
    best_angle = find_best_angle(img, offset, start_angle_range, 0.1, result_cache={})

    # print "Angle: ", best_angle - offset
    # print "Time:  ", time.time() - start_time

    return rotate(org_img, best_angle+-offset), (best_angle - offset)

#https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py
def rotate(image, angle, center=None, scale=1.0):
    rotated = misc.imrotate(image, angle)
    return rotated

def find_best_angle(img, angle_center, angle_range, angle_step_size, trim_percent=0.2, result_cache={}):
    h, w = img.shape[:2]
    eval_h = int(h * trim_percent / 2.0)
    eval_w = int(w * trim_percent / 2.0)

    max_data = (None, -float('inf'))
    # vis_vars = []
    for angle in np.arange(angle_center-angle_range, angle_center+angle_range+angle_step_size, angle_step_size):
        if angle in result_cache:
            var = result_cache[angle]
        else:
            rot_img = rotate(img, angle)[eval_h:-eval_h, eval_w:-eval_w]
            var = profile_variance(rot_img)
            result_cache[angle] = var

        # vis_vars.append(var)
        max_data = max(max_data, (angle, var), key=lambda x: x[1])
    # plt.plot(vis_vars)
    # plt.show()
    return max_data[0]

def profile_variance(img):
    #
    # sum_profile = img.sum(axis=1)
    # l = int(len(sum_profile) * 0.1)
    # return np.mean([sum_profile[:l].var(), sum_profile[-l:].var()])
    return img.sum(axis=1).var()

if __name__=="__main__":
    in_folder = sys.argv[1]
    out_folder = sys.argv[2]
    print(out_folder)

    files = [f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, f))]
    files = [f for f in files if f.endswith(".jpg") or f.endswith('.png')]
    for i, f in enumerate(sorted(files)):
        print(i, "/", len(files))
        img_path = os.path.join(in_folder, f)
        try:
            img = misc.imread(img_path)
        except:
            continue

        res, best_angle = deskew(img)

        f = f.replace(".jpg", ".png")
        misc.imsave(os.path.join(out_folder, f), res)
