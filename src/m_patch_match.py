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
			self.D = lambda x, y : np.linalg.norm(x - y)


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
		nnf = (U - xy).astype(int)

		# a little trick, set nnf's border to zero
		nnf[:self.border_size, :] = 0
		nnf[-self.border_size:, :] = 0
		nnf[:, -self.border_size:] = 0
		nnf[:, :self.border_size] = 0

		return nnf


	def __optimal_first_order_predictor(self, i, j, iter, nnf):
		'''returns the optimal first order predictor'''
		if iter % 2 == 0:
			phi1 = 2 * nnf[i - 1, j, :] - nnf[i - 2, j, :]
			phi2 = 2 * nnf[i, j - 1, :] - nnf[i, j - 2, :]
			dist1 = self.D(self.__get(np.array([i, j])), self.__get(np.array([i, j]) + phi1))
			dist2 = self.D(self.__get(np.array([i, j])), self.__get(np.array([i, j]) + phi2))
			if dist1 <= dist2:
				return phi1
			else:
				return phi2
		else:
			phi1 = 2 * nnf[i + 1, j, :] - nnf[i + 2, j, :]
			phi2 = 2 * nnf[i, j + 1, :] - nnf[i, j + 2, :]
			dist1 = self.D(self.__get(np.array([i, j])), self.__get(np.array([i, j]) + phi1))
			dist2 = self.D(self.__get(np.array([i, j])), self.__get(np.array([i, j]) + phi2))
			if dist1 <= dist2:
				return phi1
			else:
				return phi2


	def __propagation(self, nnf, iter):
		if iter % 2 == 0:
			for i in range(2, self.h):
				for j in range(2, self.w):
					nnf[i, j, :] = self.__optimal_first_order_predictor(i, j, iter, nnf)
		else:
			for i in reversed(range(0, self.h - 2)):
				for j in reversed(range(0, self.w - 2)):
					nnf[i, j, :] = self.__optimal_first_order_predictor(i, j, iter, nnf)
		return nnf


	def __random_search(self, nnf):
		return nnf





