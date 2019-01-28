import os
import sys
import json
import numpy as np
import cv2
import yaml

blue = (255,0,0)
red = (0,0,255)
green = (0,255,0)
diam = 10

def get_corners(obj, img_path=None):

    rows = []
    cols = []
    for pobj in obj['fieldBBs']:
        if pobj['type'] == 'fieldRow':
            rows.append(pobj['poly_points'])
        elif pobj['type'] == 'fieldCol':
            cols.append(pobj['poly_points'])

    if img_path is not None:
        if not os.path.exists("visuals"):
            os.makedirs("visuals")
        c_img = cv2.imread(img_path, 1)

    row_num = len(rows) + 1
    col_num = len(cols) + 1

    rowlines = np.zeros((row_num, 2, 2), dtype=int)
    for r in range(row_num):
        if r == 0:
            rowlines[r][0][0] = rows[r][0][0]
            rowlines[r][0][1] = rows[r][0][1]
            rowlines[r][1][0] = rows[r][1][0]
            rowlines[r][1][1] = rows[r][1][1]
        elif r == len(rows):
            rowlines[r][0][0] = rows[r-1][3][0]
            rowlines[r][0][1] = rows[r-1][3][1]
            rowlines[r][1][0] = rows[r-1][2][0]
            rowlines[r][1][1] = rows[r-1][2][1]
        else:
            if abs(rows[r][0][1] - rows[r-1][3][1]) < 7 and abs(rows[r][1][1] - rows[r-1][2][1]) < 7:
                rowlines[r][0][0] = (rows[r][0][0] + rows[r-1][3][0])//2
                rowlines[r][0][1] = (rows[r][0][1] + rows[r-1][3][1])//2
                rowlines[r][1][0] = (rows[r][1][0] + rows[r-1][2][0])//2
                rowlines[r][1][1] = (rows[r][1][1] + rows[r-1][2][1])//2
            else:
                rowlines[r][0][0] = rows[r][0][0]
                rowlines[r][0][1] = rows[r][0][1]
                rowlines[r][1][0] = rows[r][1][0]
                rowlines[r][1][1] = rows[r][1][1]

        if img_path is not None:
            # print(rowlines[r])
            cv2.circle(c_img, (rowlines[r,0,0], rowlines[r,0,1]), diam+5, blue, 2)
            cv2.circle(c_img, (rowlines[r,1,0], rowlines[r,1,1]), diam+5, blue, 2)

    collines = np.zeros((col_num, 2, 2), dtype=int)
    for c in range(col_num):
        if c == 0:
            collines[c][0][0] = cols[c][0][0]
            collines[c][0][1] = cols[c][0][1]
            collines[c][1][0] = cols[c][3][0]
            collines[c][1][1] = cols[c][3][1]
        elif c == len(cols):
            collines[c][0][0] = cols[c-1][1][0]
            collines[c][0][1] = cols[c-1][1][1]
            collines[c][1][0] = cols[c-1][2][0]
            collines[c][1][1] = cols[c-1][2][1]
        else:
            if abs(cols[c][0][0] - cols[c-1][1][0]) < 7 and abs(cols[c][3][0] - cols[c-1][2][0]) < 7:
                collines[c][0][0] = (cols[c][0][0] + cols[c-1][1][0])//2
                collines[c][0][1] = (cols[c][0][1] + cols[c-1][1][1])//2
                collines[c][1][0] = (cols[c][3][0] + cols[c-1][2][0])//2
                collines[c][1][1] = (cols[c][3][1] + cols[c-1][2][1])//2
            else:
                collines[c][0][0] = cols[c][0][0]
                collines[c][0][1] = cols[c][0][1]
                collines[c][1][0] = cols[c][3][0]
                collines[c][1][1] = cols[c][3][1]

        if img_path is not None:
            # print(collines[c])
            cv2.circle(c_img, (collines[c,0,0], collines[c,0,1]), diam+5, green, 2)
            cv2.circle(c_img, (collines[c,1,0], collines[c,1,1]), diam+5, green, 2)

    # rowlines[row, left, y] or rowlines[row, right, x], etc.
    # ie. rowlines[row, side, axis]
    corners = np.zeros((row_num, col_num, 2), dtype=int)
    for r in range(row_num):
        for c in range(col_num):
            # handle the corner cases
            if r == 0 and c == 0:
                corners[r,c,0] = (collines[c,0,0] + rowlines[r,0,0]) / 2
                corners[r,c,1] = (collines[c,0,1] + rowlines[r,0,1]) / 2
            elif r == len(rows) and c == len(cols):
                corners[r,c,0] = (collines[c,1,0] + rowlines[r,1,0]) / 2
                corners[r,c,1] = (collines[c,1,1] + rowlines[r,1,1]) / 2
            elif r == 0 and c == len(cols):
                corners[r,c,0] = (collines[c,0,0] + rowlines[r,1,0]) / 2
                corners[r,c,1] = (collines[c,0,1] + rowlines[r,1,1]) / 2
            elif r == len(rows) and c == 0:
                corners[r,c,0] = (collines[c,1,0] + rowlines[r,0,0]) / 2
                corners[r,c,1] = (collines[c,1,1] + rowlines[r,0,1]) / 2
            else:
                x = collines[c,0,0]
                y = rowlines[r,0,1]
                # if the rowline is sloped
                if abs(rowlines[r,0,1] - rowlines[r,1,1]) > 4:
                    x1 = rowlines[r,0,0]
                    x2 = rowlines[r,1,0]
                    y1 = rowlines[r,0,1]
                    y2 = rowlines[r,1,1]
                    slope = float(y2 - y1) / float(x2 - x1)
                    y = slope * (x - x1) + y1
                # if the colline is sloped
                if abs(collines[c,0,0] - collines[c,1,0]) > 4:
                    x1 = collines[c,0,0]
                    x2 = collines[c,1,0]
                    y1 = collines[c,0,1]
                    y2 = collines[c,1,1]
                    slope = float(x2 - x1) / float(y2 - y1)
                    x = slope * (y - y1) + x1
                corners[r,c,0] = int(x)
                corners[r,c,1] = int(y)

    if img_path is not None:
        for i in range(corners.shape[0]):
            for j in range(corners.shape[1]):
                cv2.circle(c_img, (corners[i,j,0], corners[i,j,1]), diam, red, 2)
        cv2.imwrite("visuals/{}".format(img_path.split('/')[-1]), c_img)

    return corners


def main():
    jsonfil = sys.argv[1]
    with open(jsonfil, 'r') as f:
        obj = json.load(f)
    if obj is None:
        print('None Object')
        return

    form_path = jsonfil.split('/')[0]
    trains_path = os.path.join(form_path, 'trains')
    if not os.path.exists(trains_path):
        os.makedirs(trains_path)
    # print(form_path)
    # print(trains_path)
    num = jsonfil.split('/')[-1].split('.')[0].split('template')[-1]
    # print(num)
    imgf = os.path.join(form_path, num, obj['imageFilename'])
    corners = get_corners(obj, imgf)
    temp = {'corners': corners.reshape(corners.shape[0]*corners.shape[1], corners.shape[2]).tolist()}
    with open('{}.json'.format(os.path.join(form_path, num, obj['imageFilename'].split('.')[0])), 'w') as f:
        json.dump(temp, f)

    # cpath = os.path.join('/run/user/1000/gvfs/sftp:host=ironsides,user=guest/home/guest/forms', num, obj['imageFilename'])

    temp = [ [ '{}.json'.format(os.path.join(num, obj['imageFilename'].split('.')[0])),
                os.path.join(num, obj['imageFilename']) ] ]
    tdatap = os.path.join(trains_path, num + '_tdata.json')
    with open(tdatap, 'w') as f:
        json.dump(temp, f)

    vdatap = os.path.join(trains_path, num + '_vdata.json')
    vdata = [[os.path.join(num, f.split('.')[0] + '.json'), os.path.join(num, f)]
        for f in os.listdir(os.path.join(form_path, num)) if '.jpg' in f and f != obj['imageFilename']]
    with open(vdatap, 'w') as f:
        json.dump(vdata, f)

    for v in vdata:
        if not os.path.exists(os.path.join(form_path, v[0])):
            continue
        with open(os.path.join(form_path, v[0]), 'r') as f:
            obj = json.load(f)
        if obj is None:
            print(v)
            continue
        if 'corners' in obj:
            continue
        # imgf = os.path.join(form_path, v[1])
        corners = get_corners(obj)
        temp = {'corners': corners.reshape(corners.shape[0]*corners.shape[1], corners.shape[2]).tolist()}
        with open(os.path.join(form_path, v[0]), 'w') as f:
            json.dump(temp, f)


    with open('configs/config_sample.yaml', 'r') as f:
        config = yaml.load(f)

    config['pretraining']['training_set']['img_folder'] = form_path
    config['pretraining']['training_set']['json_folder'] = form_path
    config['pretraining']['training_set']['file_list'] = tdatap
    config['pretraining']['validation_set']['img_folder'] = form_path
    config['pretraining']['validation_set']['json_folder'] = form_path
    config['pretraining']['validation_set']['file_list'] = vdatap

    config['pretraining']['snapshot_path'] = 'data/snapshots_{}/init'.format(num)

    config['evaluation']['output_path'] = '{}_segments.tsv'.format(num)
    config['evaluation']['columns'] = corners.shape[1]
    config['evaluation']['rows'] = corners.shape[0]

    with open('configs/config_{}.yaml'.format(num), 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    main()
