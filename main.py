from autoAttendance import *
import sys
from os import listdir
from PIL import Image

'''
Arguments:
- Date of rehearsal (YYYY-MM-DD) (set to "today" for today's date)
- Path to images, should be a directory containing images named {date}-{section}-{number}.png or {date}-{section}.png
- Path to CSVs of current sheets, should be a directory containing files named {date}-{section}.csv
- Number of sections to look for in images
- Number of sections to assume is always present (i.e. conductors)
- Sections to look for in images
- Sections to assume present
'''

# Extract arguments
rehearsalDate = np.datetime64('today', 'D') if sys.argv[1] == "today" else sys.argv[1]
imgPath = sys.argv[2]
csvPath = sys.argv[3]
numImgSections = int(sys.argv[4])
numPresentSections = int(sys.argv[5])
imgSections = sys.argv[6:(6 + numImgSections)]
presentSections = sys.argv[(6 + numImgSections):]

# Holds DataFrames which correspond to the section attendance sheets
filledSheets = []

for section in imgSections:
	# Stores the data read from each page
	sectionFilledCells = []

	for i, path in enumerate([file for file in listdir(imgPath) if section in file and file[-4:] == ".png"]):
		# Load image(s)
		image = np.asarray(Image.open(f"{imgPath}/{path}").convert("L"))
		image = 1 - np.int32(image > 100)

		# Extract information for this page
		rows, cols = getCoordsRowsCols(image, True, True, figTitle = f"{section}-{i}")

		# Get the filled cells on this page and append it to the section
		sectionFilledCells.append(findFilledCells(image, rows, cols, 1, 1, showFigs = True, figTitle = f"{section}-{i}"))

	# Flatten to get all filled cells as vector
	sectionFilledCells = np.array(sectionFilledCells).flatten()

	# Load a blank sheet and fill it in
	try:
		sheet = pd.read_csv(f"{csvPath}/{rehearsalDate}-{section}.csv", index_col = "ID")
		sheet["Present"] = sectionFilledCells[1:]
	except ValueError:
		print(f"***{section}-{i} failed. Defaulting to N/A.***")
		sheet["Present"] = ["N/A"] * len(sheet)
	finally:
		filledSheets.append(sheet)

for section in presentSections:
	# Load the blank sheet and fill it in with all present
	blankSheet = pd.read_csv(f"{csvPath}/{rehearsalDate}-{section}.csv", index_col = "ID")
	blankSheet["Present"] = "Y"
	filledSheets.append(blankSheet)

# Generate main sheet and write to file
mainSheet = fillOutMain(filledSheets)
mainSheet.to_csv(f"{rehearsalDate}-Attendance.csv")