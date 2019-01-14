# -*- coding: utf-8 -*-
""" Modified Patch Match algorithm
Author: Haozhe Sun
"""

import numpy as np
from time import time
from m_patch_match import M_PatchMatch
from PIL import Image
import os
from utils import *
import argparse


t0 = time()

root_list = ["original", "modified"]

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, default="train.jpg")
parser.add_argument("--root", type=str, default="modified")
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--nnf_threshold", type=int, default=10)
parser.add_argument("--iter", type=int, default=5)
parser.add_argument("--binary_threshold", type=float, default=0.3)
args = parser.parse_args()

root = args.root
filename = args.filename

assert root in root_list, "Invalid root argument"


directory = os.path.join("../data", root)

if root == "modified":
	name, jpg = filename.split(".")
	filename = name + "_." + jpg

path = os.path.join(directory, filename)

img = Image.open(path)
I = np.asarray(img)
mpm = M_PatchMatch(I, patch_size=args.patch_size, get_dist=True)


nnf, df = mpm.run(nb_iter=args.iter)

# post-processing
nnf, df, binary_map = filtering(args.nnf_threshold, nnf, df, args.binary_threshold)


print(nnf.shape)
print("Done in %.4f s." % (time() - t0))

visualize(nnf, df, binary_map, img, filename)
