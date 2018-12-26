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
mpm = M_PatchMatch(I)

#mpm = M_PatchMatch(np.random.uniform(size=(768, 1024, 3)))
#mpm = M_PatchMatch(np.ones((4, 5)), patch_size=1)
nnf = mpm.run()

print(nnf.shape)
print("\n")
print(nnf[:, :, 0])
print("\n")
print(nnf[:, :, 1])

print("Done in %.4f s." % (time() - t0))