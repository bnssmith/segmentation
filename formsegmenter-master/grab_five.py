import os
import sys
import json
from os.path import join as opjoin
from random import shuffle
from shutil import copyfile


def main():
    gfpath = sys.argv[1]
    dirpath = sys.argv[2]

    with open(gfpath, 'r') as f:
        lines = [line.split(',') for line in f]

    for line in lines:
        if int(line[1]) > 4:
            if not os.path.exists(opjoin('temp_forms', line[0])):
                os.makedirs(opjoin('temp_forms', line[0]))
                lst = [f for f in os.listdir(opjoin(dirpath, line[0])) if '.jpg' in f]
                shuffle(lst)
                for f in lst[:5]:
                    # print(opjoin(dirpath, line[0], f))
                    copyfile(opjoin(dirpath, line[0], f), opjoin('temp_forms', line[0], f))

                for temp in os.listdir(opjoin(dirpath, line[0])):
                    if 'template' in temp:
                        copyfile(opjoin(dirpath, line[0], temp), opjoin('temp_forms/trains', line[0], temp))

                print(line[0])


if __name__ == '__main__':
    main()
