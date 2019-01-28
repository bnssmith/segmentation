import os
import sys
import json
import numpy as np
import cv2
import yaml
import math

blue = (255,0,0)
red = (0,0,255)
green = (0,255,0)
diam = 10

def euc_dist(p1, p2):
    return math.sqrt( pow(p2[1] - p1[1], 2) / (pow(p2[0] - p1[0], 2) + 0.0001) )

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

    dist_thresh = 20.0
    # rowlines = np.zeros((row_num, 2, 2), dtype=int)
    rowlines = []
    for r in range(len(rows)):
        # topleft
        x = rows[r][0][0]
        y = rows[r][0][1]
        left = (x, y)
        # topright
        x = rows[r][1][0]
        y = rows[r][1][1]
        right = (x, y)
        rowlines.append( (left, right) )
        # botleft
        x = rows[r-1][3][0]
        y = rows[r-1][3][1]
        left = (x, y)
        # botright
        x = rows[r-1][2][0]
        y = rows[r-1][2][1]
        right = (x, y)
        rowlines.append( (left, right) )

    def sort_ykey(elem):
        return elem[0][1]
    rowlines.sort(key=sort_ykey)
    # rowlines = np.asarray(rowlines)
    # print(rowlines.shape)
    # for r in range(rowlines.shape[0]-1):
    #     if r >= rowlines.shape[0]-1:
    #         break
    print(len(rowlines))
    for r in range(len(rowlines)-1):
        if r >= len(rowlines)-1:
            break
        """
        if the left of the current rowline and
        the left of the prev rowline are within dist_thresh,
        we take the avg of them to make our new point
        (note: we check the rights as well)"""
        # if euc_dist(rowlines[r][0], rowlines[r+1][0]) < dist_thresh \
        #     and euc_dist(rowlines[r][1], rowlines[r+1][1]) < dist_thresh:
        # if np.linalg.norm(np.asarray(rowlines[r][0]) - np.asarray(rowlines[r+1][0])) < dist_thresh \
        #     and np.linalg.norm(np.asarray(rowlines[r][1]) - np.asarray(rowlines[r+1][1])):
        if abs(rowlines[r][0][1] - rowlines[r+1][0][1]) < dist_thresh:
            # left
            x = (rowlines[r][0][0] + rowlines[r+1][0][0])//2
            y = (rowlines[r][0][1] + rowlines[r+1][0][1])//2
            left = (x, y)
            # right
            x = (rowlines[r][1][0] + rowlines[r+1][1][0])//2
            y = (rowlines[r][1][1] + rowlines[r+1][1][1])//2
            right = (x, y)
            rowlines[r] = (left, right)
            del rowlines[r+1]
        else:
            print( 'pt1: {}, pt2: {}, pt3: {}, pt4: {}'.format(
                rowlines[r][0], rowlines[r+1][0],
                rowlines[r][1], rowlines[r+1][1]) )
    print(len(rowlines))

    if img_path is not None:
        for r in range(len(rowlines)):
            cv2.circle(c_img, (rowlines[r][0][0], rowlines[r][0][1]), diam+5, blue, 2)
            cv2.circle(c_img, (rowlines[r][1][0], rowlines[r][1][1]), diam+5, blue, 2)
    rowlines = np.asarray(rowlines)

    # collines = np.zeros((col_num, 2, 2), dtype=int)
    collines = []
    for c in range(len(cols)):
        x = cols[c][0][0]
        y = cols[c][0][1]
        top = (x, y)
        x = cols[c][3][0]
        y = cols[c][3][1]
        bot = (x, y)
        collines.append( (top, bot) )
        # topright
        x = cols[c-1][1][0]
        y = cols[c-1][1][1]
        top = (x, y)
        # botright
        x = cols[c-1][2][0]
        y = cols[c-1][2][1]
        bot = (x, y)
        collines.append( (top, bot) )

    def sort_xkey(elem):
        return elem[0][0]
    collines.sort(key=sort_xkey)

    print(len(collines))
    for c in range(len(collines)-1):
        if c >= len(collines)-1:
            break
        """
        if this is not the first col:
        if the topleft of the current col and
        the topright of the prev col are within dist_thresh,
        we take the avg of them to make our new point
        (note: we check the bottoms as well)"""
        if abs(collines[c][0][0] - collines[c+1][0][0]) < dist_thresh:
            x = (collines[c][0][0] + collines[c+1][0][0])//2
            y = (collines[c][0][1] + collines[c+1][0][1])//2
            top = (x, y)
            x = (collines[c][1][0] + collines[c+1][1][0])//2
            y = (collines[c][1][1] + collines[c+1][1][1])//2
            bot = (x, y)
            collines[c] = (top, bot)
            del collines[c+1]
        else:
            print( 'pt1: {}, pt2: {}, pt3: {}, pt4: {}'.format(
                collines[c][0], collines[c+1][0],
                collines[c][1], collines[c+1][1]) )
    print(len(collines))

    if img_path is not None:
        for c in range(len(collines)):
            cv2.circle(c_img, (collines[c][0][0], collines[c][0][1]), diam+5, green, 2)
            cv2.circle(c_img, (collines[c][1][0], collines[c][1][1]), diam+5, green, 2)
    collines = np.asarray(collines)

    # rowlines[row, left, y] or rowlines[row, right, x], etc.
    # ie. rowlines[row, side, axis]
    comp_thresh = 50
    row_num = rowlines.shape[0]
    col_num = collines.shape[0]
    corners = np.zeros((row_num, col_num, 2), dtype=int)
    for r in range(row_num):
        row_left_x = rowlines[r,0,0]
        row_right_x = rowlines[r,1,0]
        row_comp_y = (rowlines[r,0,1] + rowlines[r,1,1]) / 2.0
        for c in range(col_num):
            col_comp_x = (collines[c,0,0] + collines[c,1,0]) / 2.0
            col_top_y = collines[c,0,1]
            col_bot_y = collines[c,1,1]
            """check to see if the colline is within the x bounds of the row
            and to see if the rowline is within the y bounds of the col"""
            if (row_left_x - comp_thresh) < col_comp_x \
                    and (row_right_x + comp_thresh) > col_comp_x \
                    and (col_top_y - comp_thresh) < row_comp_y \
                    and (col_bot_y + comp_thresh) > row_comp_y:

                y = row_comp_y
                x = col_comp_x
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
            # else:
            #     print( 'row_comp_y: {} < {} < {}'.format(
            #         (col_top_y - comp_thresh), row_comp_y, (col_bot_y + comp_thresh)) )
            #     print( 'col_comp_x: {} < {} < {}'.format(
            #         (row_left_x - comp_thresh), col_comp_x, (row_right_x + comp_thresh)) )

    if img_path is not None:
        for i in range(corners.shape[0]):
            for j in range(corners.shape[1]):
                cv2.circle(c_img, (corners[i,j,0], corners[i,j,1]), diam, red, 2)
        cv2.imwrite("visuals/{}".format(img_path.split('/')[-1]), c_img)

    return corners


def main():
    """expecting the template file"""
    jsonfil = sys.argv[1]
    with open(jsonfil, 'r') as f:
        obj = json.load(f)
    if obj is None:
        print('None Object')
        return

    form_path = '/'.join(jsonfil.split('/')[:-2])
    trains_path = os.path.join(form_path, 'trains')
    if not os.path.exists(trains_path):
        os.makedirs(trains_path)
    print(form_path)
    print(trains_path)
    num = jsonfil.split('/')[-1].split('.')[0].split('template')[-1]
    # print(num)
    imgf = os.path.join(form_path, num, obj['imageFilename'])
    corners = get_corners(obj, imgf)
    temp = {'corners': corners.reshape(corners.shape[0]*corners.shape[1], corners.shape[2]).tolist()}
    tempjson = os.path.join(form_path, num, obj['imageFilename'].split('.')[0]) + '.json'
    with open(tempjson, 'w') as f:
        json.dump(temp, f)

    temp = [ [ '{}.json'.format(os.path.join(num, obj['imageFilename'].split('.')[0])),
                os.path.join(num, obj['imageFilename']) ] ]
    tdatap = os.path.join(trains_path, num + '_tdata.json')
    with open(tdatap, 'w') as f:
        json.dump(temp, f)

    vdatap = os.path.join(trains_path, num + '_vdata.json')
    vdata = [[os.path.join(num, f.split('.')[0] + '.json'), os.path.join(num, f)]
        for f in os.listdir(os.path.join(form_path, num))
        if '.jpg' in f and f != obj['imageFilename']
        and os.path.isfile(os.path.join(num, f.split('.')[0] + '.json'))]
    # print(vdata)
    with open(vdatap, 'w') as f:
        json.dump(vdata, f)

    for v in vdata:
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

    config['pretraining']['snapshot_path'] = 'snapshots/snapshots_{}/init'.format(num)

    config['evaluation']['output_path'] = '{}_segments.tsv'.format(num)
    config['evaluation']['columns'] = corners.shape[1]
    config['evaluation']['rows'] = corners.shape[0]

    with open('configs/config_{}.yaml'.format(num), 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    main()
