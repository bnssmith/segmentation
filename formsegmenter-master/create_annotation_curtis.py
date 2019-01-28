import cv2
import numpy as np
import sys
import json

def connect_components(img):

    connectivity = 4
    output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    return output

img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)

vert = img[...,0]
horz = img[...,2]

overlap = np.logical_and(vert, horz)


centroids = connect_components(overlap.astype(np.uint8))[3]


corners = centroids.tolist()
with open(sys.argv[2], 'w') as f:
    json.dump({
        "corners": corners
    }, f)
