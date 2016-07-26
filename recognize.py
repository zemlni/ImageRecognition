from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
#import time

xValues = []
yValues = []

def threshold(imageArray):
	balanceArray = []
	for row in imageArray:
		for pixel in row:
			average = getAverage(pixel)
			balanceArray.append(average)
	balance = 0
	for num in balanceArray:
		balance += num
	balance = balance / len(balanceArray)
	answer = list(imageArray)
	for row in answer:
		for pixel in row:
			if len(pixel) == 4:
				pixel[3] = 255
		#	print pixel
			testAverage = getAverage(pixel)
                	if testAverage > balance:
                		pixel[0] = 255
		                pixel[1] = 255
		                pixel[2] = 255
	                else:
            		        pixel[0] = 0
                                pixel[1] = 0
               			pixel[2] = 0
	return answer
def getAverage(pixel):
	average = 0
	for num in pixel:
		average += num
	return average / len(pixel[:3])

def getColor(pixel):
	if pixel[0] == pixel[1] and pixel[1] == pixel[2]:
		if pixel[0] == 255 : return "white"
		if pixel[0] == 0: return "black"
	return "ERROR"

def generatePolynomial(imageArray):
	radius = 5 
	lastPoint = (0, 0)
	background = getColor(imageArray[0][0])
	for row in range(0, len(imageArray)):
		for col in range(0, len(imageArray[0])):
			pixel = imageArray[row][col]
			#print(pixel)
			if getColor(pixel) != background:
				#print(getColor(pixel))
				#print("row: " + str(row) + "col: " + str(col) + "distance: " + str((((row - lastPoint[0]) ** 2 + (col - lastPoint[1]) ** 2) ** (.5)))+ "last row: " + str(lastPoint[0]) + "last col: " + str(lastPoint[1]))
				if (((row - lastPoint[0]) ** 2 + (col - lastPoint[1]) ** 2) ** (.5)) > radius:
					#print("test")
					xValues.append(row)
					yValues.append(col)
					lastPoint = (row, col)
	#print(xValues)
	#print(yValues)
	polynomial, stats = np.polynomial.polynomial.polyfit(xValues, yValues, 0, full=True)
	for i in range(0, 10):
		cur, curStats = np.polynomial.polynomial.polyfit(xValues, yValues, i, full=True)
		if stats[0] > curStats[0]:
			polynomial = cur
			stats = curStats
	return polynomial	

image = Image.open('images/line2.png')
width, height = image.size
#test = image.load()
#print(test)
#print(test[0, 0])
imageArray = np.array(image)
#print(imageArray)
thresholdedImageArray = threshold(imageArray)
polynomial = generatePolynomial(thresholdedImageArray)
u = np.linspace(0, width, 100)
v = np.polynomial.polynomial.polyval(u, polynomial)
plt.figure()
plt.ylim(0, height)
plt.plot(xValues, yValues, ".")
plt.plot(u, v, "--")
plt.show()

