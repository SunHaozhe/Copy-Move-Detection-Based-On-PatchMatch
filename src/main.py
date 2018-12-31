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


t0 = time()

filename = "train.jpg"

original_dir = "../data/original"
modified_dir = "../data/modified"
original_path = os.path.join(original_dir, filename)
name, jpg = filename.split(".")
modified_path = os.path.join(modified_dir, name + "_." + jpg )


img = Image.open(modified_path)
I = np.asarray(img)
mpm = M_PatchMatch(I, patch_size=10, get_dist=True)

#mpm = M_PatchMatch(np.random.uniform(size=(768, 1024, 3)))
#mpm = M_PatchMatch(np.ones((7, 9)), patch_size=1, border_size=1)
nnf, df = mpm.run(nb_iter=5)

# post-processing
thr = 20
nnf, df = filtering(thr, nnf, df)


print(nnf.shape)
print("Done in %.4f s." % (time() - t0))

visualize(nnf, df, img, filename)
