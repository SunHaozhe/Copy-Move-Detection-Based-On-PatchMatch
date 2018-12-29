import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_nnf(nnf, df):
	plt.figure(figsize=(12, 6.5))
	plt.suptitle("Nearest Neighbor Field map (absolute value)")

	ax = plt.subplot(221)
	x = np.linspace(0, nnf[:, :, 0].shape[1], num=nnf[:, :, 0].shape[1]) 
	y = np.linspace(0, nnf[:, :, 0].shape[0], num=nnf[:, :, 0].shape[0])
	X, Y = np.meshgrid(x , y) 
	plt.pcolormesh(X, Y, np.abs(nnf[:, :, 0]), cmap = cm.copper) 
	plt.colorbar()
	ax.set_title("First dimension (axis)")

	ax = plt.subplot(222)
	plt.pcolormesh(X, Y, np.abs(nnf[:, :, 1]), cmap = cm.copper) 
	plt.colorbar()
	ax.set_title("Second dimension (axis)")

	ax = plt.subplot(223)
	plt.pcolormesh(X, Y, 
				   np.abs(nnf[:, :, 0]) + np.abs(nnf[:, :, 1]), 
				   cmap = cm.copper) 
	plt.colorbar()
	ax.set_title("L1 norm of NNF map")

	ax = plt.subplot(224)
	plt.pcolormesh(X, Y, df[:, :], cmap = cm.copper) 
	plt.colorbar()
	ax.set_title("Distance field map")

	plt.show()