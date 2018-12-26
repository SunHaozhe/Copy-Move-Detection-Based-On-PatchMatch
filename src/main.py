# -*- coding: utf-8 -*-
""" Modified Patch Match algorithm
Author: Haozhe Sun
"""

import numpy as np
from time import time
from m_patch_match import M_PatchMatch
from PIL import Image
import os


t0 = time()

path = "../data"
filename = "cat.jpeg"

img = Image.open(os.path.join(path, filename))
I = np.asarray(img)
mpm = M_PatchMatch(I, patch_size=96)

#mpm = M_PatchMatch(np.random.uniform(size=(768, 1024, 3)))
#mpm = M_PatchMatch(np.ones((7, 9)), patch_size=1, border_size=1)
nnf = mpm.run(nb_iter=5)

print(nnf.shape)
print("\n")
print(nnf[:, :, 0])
print("\n")
print(nnf[:, :, 1])

print("Done in %.4f s." % (time() - t0))