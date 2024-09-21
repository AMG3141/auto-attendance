import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import pandas as pd

SHOW_ALL_FIGS = False # Only set this to true when you want to see the big convolutions and line plots etc. Normal showFigs will just show rows/cols

'''
Takes a binary image of a table where the table and data are non-zero and the background is zero. Returns arrays containing the coordinates of the rows and columns of the table.
'''
def getCoordsRowsCols(image, verbose = False, showFigs = False, figTitle = None):
	# Masks for convolution
	maskV = np.ones([image.shape[0], 1]) / image.shape[0]
	maskH = np.ones([1, image.shape[1]]) / image.shape[1]

	# Convolved images (basically gives the average of each row/col)
	if verbose:
		print("Performing convolutions")
		print("Vertical")
	if showFigs and SHOW_ALL_FIGS: vert = convolve2d(image, maskV) # Better for visualisation
	else:
		vert = convolve2d(image, maskV, mode = "valid") # Wayyyy more efficient
		vert = vert.T # Makes life easier

	if verbose:
		print("Horizontal")
	if showFigs and SHOW_ALL_FIGS: horiz = convolve2d(image, maskH) # Better for visualisation
	else:
		horiz = convolve2d(image, maskH, mode = "valid") # Wayyyy more efficient

	if verbose: print("Done")

	if showFigs and SHOW_ALL_FIGS:
		plt.imshow(vert, cmap = "Greys")
		plt.title(figTitle)
		plt.show()
		plt.cla()
		plt.imshow(horiz, cmap = "Greys")
		plt.title(figTitle)
		plt.show()

	# Calculate the the mean value of the convolved images by either row or column (if we didn't do the better way)
	if showFigs and SHOW_ALL_FIGS:
		vert = np.mean(vert, axis = 0)
		horiz = np.mean(horiz, axis = 1)

	if showFigs and SHOW_ALL_FIGS:
		plt.subplot(1, 2, 1)
		plt.plot(vert, label = "Vertical")
		plt.hlines(np.mean(vert), 0, len(vert), "r", label = "$\mu$")
		plt.hlines(np.mean(vert) + 5 * np.std(vert), 0, len(vert), "k", label = "$\mu+4\sigma$")
		plt.title(figTitle + " Cols")
		plt.legend()

		plt.subplot(1, 2, 2)
		plt.plot(horiz, label = "Horizontal")
		plt.hlines(np.mean(horiz), 0, len(horiz), "r", label = "$\mu$")
		plt.hlines(np.mean(horiz) + 3 * np.std(horiz), 0, len(horiz), "k", label = "$\mu+3\sigma$")
		plt.title(figTitle + " Rows")
		plt.legend()
		plt.show()

	# Find the indices of the values which are statistical outliers
	# Using sample mean and standard deviation, so outliers will have a large influence, but it doesn't really matter for this application
	# We define outliers by any value which is more than a given number of standard deviation from the mean, 5 for vertical, 3 for horizontal (these numbers found through trial and error)
	if verbose: print("Finding outliers")
	vertOutliers = np.where(vert > np.mean(vert) + 5 * np.std(vert))[0]
	horizOutliers = np.where(horiz > np.mean(horiz) + 3 * np.std(horiz))[0]

	# Remove any consecutive indices
	badVertIndices, badHorizIndices = [], []
	for i in range(len(vertOutliers) - 1):
		if vertOutliers[i] + 1 == vertOutliers[i + 1]:
			badVertIndices.append(i)
	vertOutliers = np.delete(vertOutliers, badVertIndices)

	for i in range(len(horizOutliers) - 1):
		if horizOutliers[i] + 1 == horizOutliers[i + 1]:
			badHorizIndices.append(i)
	horizOutliers = np.delete(horizOutliers, badHorizIndices)
	if verbose: print("Done")

	if showFigs:
		plt.imshow(image, cmap = "Greys")
		plt.vlines(vertOutliers, 0, image.shape[0] - 1)
		plt.hlines(horizOutliers, 0, image.shape[1] - 1)
		plt.title(figTitle)
		plt.show()

	return horizOutliers, vertOutliers

# Go through an image using the given rows and columns and look for filled in cells in rows lower than or equal to the start row and in the column col
def findFilledCells(image, rows, cols, startRow, col, threshold = 0.005, buffer = 15, verbose = False, showFigs = False, figTitle = None):
	result = np.array([""] * (len(rows) - 1))# np.zeros(len(rows) - 1)
	for i in range(startRow, len(result)):
		try:
			left = rows[i] + buffer
			right = rows[i + 1] - buffer
			top = cols[col] + buffer
			bottom = cols[col + 1] - buffer
		except IndexError:
			print(f"***{figTitle} failed. Defaulting to N/A.***")
			return ["N/A"] * len(rows)

		if np.sum(image[left:right, top:bottom]) / ((right - left) * (bottom - top)) > threshold:
			if verbose: print(f"Value found in row {i} ({np.sum(image[left:right, top:bottom])}, {((right - left) * (bottom - top))})")
			if showFigs: image[left:right, top:bottom] += 1
			result[i] = "Y"
		else:
			if verbose: print(f"Value not found in row {i} ({np.sum(image[left:right, top:bottom])}, {((right - left) * (bottom - top))})")
			result[i] = "N"

	if showFigs:
		plt.imshow(image, cmap = "Greys")
		plt.colorbar()
		plt.title(figTitle)
		plt.show()

	return result

# Combine all the section sheets and sort them by ID
def fillOutMain(filledSheets):
	mainSheet = pd.concat(filledSheets)
	mainSheet.sort_values(by = "ID", inplace = True)
	return mainSheet