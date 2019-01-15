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

	def __init__(self, I, patch_size=16, D="l1", border_size=0, 
				 non_zero_nnf=True, get_dist=True):
		'''
		I: image, numpy array
		D: distance measure, default is L1 distance
		patch_size: the size of patch, int
		border_size: width of border that will be initialized by 0
		'''
		self.I = I
		self.patch_size = patch_size
		self.border_size = border_size
		self.non_zero_nnf = non_zero_nnf
		self.get_dist = get_dist

		# height of available regions of the reprensentative pixels of patches
		self.h = self.I.shape[0] - patch_size + 1 
		# width of available regions of the representative pixels of patches
		self.w = self.I.shape[1] - patch_size + 1 

		assert self.h > 0 and self.w > 0, "patch_size is too large for this image"
		assert D == "l1", "Only L1 norm distance measure is supported"
		assert border_size >= 0 and border_size <= 1, \
			   "border_size can be either 0 or 1, not others"

		if D == "l1":
			self.D = lambda x, y : np.sum(np.abs(x - y))

		grid = [- 1, 0, 1]
		self.R_space = []

		for i in range(len(grid)):
			for j in range(len(grid)):
				if grid[i] == 0 and grid[j] == 0:
					continue
				self.R_space.append(np.array([grid[i], grid[j]]))
		self.R_space = np.array(self.R_space)


	def run(self, nb_iter=3):
		'''
		nb_iter: number of iterations of the algorithm
		'''
		nnf = self.__initialize()
		for iter in range(nb_iter):
			nnf = self.__propagation(nnf, iter)
			nnf = self.__random_search(nnf, iter)

		df = np.zeros((nnf.shape[0], nnf.shape[1]))
		if self.get_dist:
			for i in range(self.h):
				for j in range(self.w):
					df[i, j] = self.D(self.__get(np.array([i, j])), 
										   self.__get(np.array([i, j]) + nnf[i, j, :]))
			return nnf, df

		return nnf


	def __get(self, z):
		'''
		gets the patch corresponding to the representative pixel z(x, y)
		'''
		assert self.__is_inside(z[0], z[1], np.array([0, 0])), \
				"The representative pixel is out of bound"
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
		
		if self.non_zero_nnf:
			for i in range(self.h):
				for j in range(self.w):
					if nnf[i, j, 0] == 0 and nnf[i, j, 1] == 0:
						if self.__is_inside(i, j, np.array([1, 1])):
							nnf[i, j, 0] = 1
							nnf[i, j, 1] = 1
						else:
							nnf[i, j, 0] = - 1
							nnf[i, j, 1] = - 1

		# a little trick, set nnf's border to zero
		if self.border_size != 0:
			nnf[:self.border_size, :] = 0
			nnf[-self.border_size:, :] = 0
			nnf[:, -self.border_size:] = 0
			nnf[:, :self.border_size] = 0

		return nnf


	def __optimal_first_order_predictor(self, i, j, iter, nnf):
		'''returns the optimal first order predictor'''
		if iter % 2 == 0:
			phi1 = 2 * nnf[i - 1, j, :] - nnf[i - 2, j, :]
			if not self.__is_inside(i, j, phi1):
				phi1 = nnf[i - 1, j, :]

			phi2 = 2 * nnf[i, j - 1, :] - nnf[i, j - 2, :]
			if not self.__is_inside(i, j, phi2):
				phi2 = nnf[i, j - 1, :]
			#assert self.__is_inside(i, j, phi1), "Not inside phi1: " + str(np.array([i, j]) + phi1) + ", " + str((self.h, self.w))
			#assert self.__is_inside(i, j, phi2), "Not inside phi2: " + str(np.array([i, j]) + phi2) + ", " + str((self.h, self.w))
		else:
			phi1 = 2 * nnf[i + 1, j, :] - nnf[i + 2, j, :]
			if not self.__is_inside(i, j, phi1):
				phi1 = nnf[i + 1, j, :]

			phi2 = 2 * nnf[i, j + 1, :] - nnf[i, j + 2, :]
			if not self.__is_inside(i, j, phi2):
				phi2 = nnf[i, j + 1, :]

		if self.__is_inside(i, j, phi1):
			dist1 = self.D(self.__get(np.array([i, j])), self.__get(np.array([i, j]) + phi1))
		else:
			dist1 = float("inf")

		if self.__is_inside(i, j, phi2):
			dist2 = self.D(self.__get(np.array([i, j])), self.__get(np.array([i, j]) + phi2))
		else:
			dist2 = float("inf")

		phi3 = nnf[i, j, :]
		dist3 = self.D(self.__get(np.array([i, j])), self.__get(np.array([i, j]) + phi3))

		if self.non_zero_nnf:
			if phi1[0] == 0 and phi1[1] == 0:
				dist1 = float("inf")
			if phi2[0] == 0 and phi2[1] == 0:
				dist2 = float("inf")

		if dist1 <= min(dist2, dist3):
			return phi1
		elif dist2 <= min(dist1, dist3):
			return phi2
		else:
			return phi3


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


	def __is_inside(self, i, j, offset):
		'''
		checks if the (pixel(i, j) + offset) is still inside the valid region of
		representative pixels
		'''
		if (i + offset[0] >= 0) and \
		   (i + offset[0] < self.h) and \
		   (j + offset[1] >= 0) and \
		   (j + offset[1] < self.w):
			return True
		else:
			return False


	def __build_candidates(self, nnf, i, j):
		'''
		finds valid candidates for random search
		'''
		ii = 1
		candidates = [nnf[i, j, :]]
		while(True):
			space = self.R_space
			space = [item for item in self.R_space \
					if self.__is_inside(i, j, nnf[i, j, :] + int(2 ** (ii - 1)) * item)]
			if len(space) == 0:
				break
			idx = np.random.choice(range(len(space)))
			R = space[idx]
			candidates.append(nnf[i, j, :] + int(2 ** (ii - 1)) * R)
			ii += 1
		return candidates


	def __random_search_for_one_pixel(self, nnf, i, j):
		candidates = self.__build_candidates(nnf, i, j)
		result = candidates[0]
		min_dist = float("inf")
		for candidate in candidates:
			dist = self.D(self.__get(np.array([i, j])), 
						  self.__get(np.array([i, j]) + candidate))
			if self.non_zero_nnf:
				new = np.array([i, j]) + candidate
				if candidate[0] == 0 and candidate[1] == 0:
					dist = float("inf")
			if dist < min_dist:
				min_dist = dist
				result = candidate
		return result


	def __random_search(self, nnf, iter):
		if iter % 2 == 0:
			for i in range(self.h):
				for j in range(self.w):
					nnf[i, j, :] = self.__random_search_for_one_pixel(nnf, i, j)
		else:
			for i in reversed(range(self.h)):
				for j in reversed(range(self.w)):
					nnf[i, j, :] = self.__random_search_for_one_pixel(nnf, i, j)
		return nnf


	def __nb_zero_offsets(self, nnf):
		return np.count_nonzero(np.bitwise_and(nnf[:, :, 0] == 0, nnf[:, :, 1] == 0))





