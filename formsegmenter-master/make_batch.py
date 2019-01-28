import os
from os.path import join
import sys

mdir = sys.argv[1]

fils = [f for f in os.listdir(mdir)]

fols = [f for f in fils if os.path.isdir(f)]
csvs = [f.split('.')[0] for f in fils if '.csv' in f]
print(csvs)
fols = [f for f in fols if f not in csvs]
print(fols)
