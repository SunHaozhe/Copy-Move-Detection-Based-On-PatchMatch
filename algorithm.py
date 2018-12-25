# -*- coding: utf-8 -*-
""" Modified Patch Match algorithm
Author: Haozhe Sun
"""

import numpy as np 

class M_PatchMatch():
	'''
	Modified Patch Match algorithm
	Outputs a Nearest Neighbor field mapping
	'''

	def __init__(self, I, patch_size=16, D="l2", border_size=2):
		'''
		I: image, numpy array
		D: distance measure, default is L2 distance
		patch_size: the size of patch, int
		border_size: width of border that will be initialized by 0
		'''
		self.I = I
		self.patch_size = patch_size
		self.border_size = border_size

		# height of available regions of the reprensentative pixels of patches
		self.h = self.I.shape[0] - patch_size + 1 
		# width of available regions of the representative pixels of patches
		self.w = self.I.shape[1] - patch_size + 1 

		assert self.h > 0 and self.w > 0, "patch_size is too large for this image"
		assert D == "l2", "Only L2 distance measure is supported"
		assert border_size >= 0

		if D == "l2":
			self.D = lambda x, y : np.linalg.norm(x - y, "fro")


	def run(self, nb_iter=5):
		'''
		nb_iter: number of iterations of the algorithm
		'''
		nnf = self.__initialize()
		for iter in range(nb_iter):
			nnf = self.__propagation(nnf, iter)
			nnf = self.__random_search(nnf)
		return nnf


	def __get(self, z):
		'''
		gets the patch corresponding to the representative pixel z(x, y)
		'''
		return self.I[z[0]:z[0] + self.patch_size, z[1]:z[1] + self.patch_size]


	def __initialize(self):
		'''
		nnf: Nearest Neighbor field, numpy array of offsets (self.h, self.w, 2)
		'''
		xy = np.zeros((self.h, self.w, 2)) 
		xy[:, :, 0], xy[:, :, 1] = np.meshgrid(range(self.h), 
			         					       range(self.w), 
			         					 	   indexing='ij')
		U = np.zeros((self.h, self.w, 2))
		U[:, :, 0] = np.random.randint(0, self.h, size=(self.h, self.w))
		U[:, :, 1] = np.random.randint(0, self.w, size=(self.h, self.w))
		nnf = U - xy

		# a little trick, set nnf's border to zero
		nnf[:self.border_size, :] = 0
		nnf[-self.border_size:, :] = 0
		nnf[:, -self.border_size:] = 0
		nnf[:, :self.border_size] = 0

		return nnf


	def __first_order_predictors(i, j, iter):
		'''returns candidate first order predictors'''
		d = {}
		return d


	def __propagation(self, nnf, iter):
		if iter % 2 == 0:
			for j in range(1, self.h):
				for i in range(1, self.w):
					d = __first_order_predictors(i, j, iter)
					nnf[i, j, :] = min(d, key=d.get)
		else:
			pass
		return nnf


	def __random_search(self, nnf):
		return nnf

mpm = M_PatchMatch(np.ones((768, 1024, 3)))
#mpm = M_PatchMatch(np.ones((3, 3)), patch_size=1)
nnf = mpm.run()
print(nnf[:, :, 0])
print("\n")
print(nnf[:, :, 1])
