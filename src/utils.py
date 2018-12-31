import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def visualize(nnf, df, img, input_filename):
	fig = plt.figure(figsize=(12, 6.5))
	plt.suptitle("Nearest Neighbor Field map (absolute value)")

	ax = plt.subplot(231)
	x = np.linspace(0, nnf[:, :, 0].shape[1], num=nnf[:, :, 0].shape[1]) 
	y = np.linspace(0, nnf[:, :, 0].shape[0], num=nnf[:, :, 0].shape[0])
	X, Y = np.meshgrid(x , y) 
	plt.pcolormesh(X, Y, np.abs(nnf[:, :, 0]), cmap = cm.copper) 
	plt.colorbar()
	plt.gca().invert_yaxis()
	ax.set_title("First dimension (axis)")

	ax = plt.subplot(232)
	plt.pcolormesh(X, Y, np.abs(nnf[:, :, 1]), cmap = cm.copper) 
	plt.colorbar()
	plt.gca().invert_yaxis()
	ax.set_title("Second dimension (axis)")

	ax = plt.subplot(233)
	plt.pcolormesh(X, Y, 
				   np.abs(nnf[:, :, 0]) + np.abs(nnf[:, :, 1]), 
				   cmap = cm.copper) 
	plt.colorbar()
	plt.gca().invert_yaxis()
	ax.set_title("L1 norm of NNF map")

	ax = plt.subplot(234)
	plt.pcolormesh(X, Y, df, cmap = cm.copper) 
	plt.colorbar()
	plt.gca().invert_yaxis()
	ax.set_title("Distance field map")

	ax = plt.subplot(235)
	plt.pcolormesh(X, Y, np.log(df), cmap = cm.copper) 
	plt.colorbar()
	plt.gca().invert_yaxis()
	ax.set_title("Distance field map in log scale")

	ax = plt.subplot(236)
	plt.imshow(img)
	ax.set_title("Image")

	dir_path = "../output"
	output_path = os.path.join(dir_path, 
							   input_filename.split(".")[0] + "_output.jpg")
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

	fig.savefig(output_path, dpi=fig.dpi)
	plt.show()


def filtering(thr, nnf, df):
	h = nnf.shape[0]
	w = nnf.shape[1]
	max_value = np.max(df)
	quantile_value = np.quantile(df, 0.5)

	for i in range(h):
		for j in range(w):
			if np.sum(np.abs(nnf[i, j, :])) < thr:
				nnf[i, j, :] = np.array([0, 0])
				df[i, j] = max_value


	for i in range(h):
		for j in range(w):
			if df[i, j] > quantile_value:
				df[i, j] = max_value

	return nnf, df