import cv2
import numpy as np

#Reading Given Satellite Image
SatelliteImage = cv2.imread('yellow_detect.jpeg')

#Changing Color From BGR to HSV
ColorBGR2HSV = cv2.cvtColor(SatelliteImage, cv2.COLOR_BGR2HSV)

#In HSV, Color range of yellow is 22 to 30
Light_Yellow = np.array([22, 103, 103])
Dark_Yellow = np.array([29, 255, 255])

#Applying yellow color range, detect yellow block
YellowDetected = cv2.inRange(ColorBGR2HSV, Light_Yellow, Dark_Yellow)

#Find Contours(...Simply Shape) of yellow block
Contours = cv2.findContours(YellowDetected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Contours = Contours[0] if len(Contours) == 2 else Contours[1]

#Draw Rectangle around Yellow Block To get its Exact Center
for contour in Contours:
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(ColorBGR2HSV, (x, y), (x + w, y + h),0,0)
    #find centre coordinate of the rectangle and print it using contour moments
    CentreMoment = cv2.moments(contour+1)
    if CentreMoment["m00"] != 0:
        X_Coordinate =int(CentreMoment["m10"] / CentreMoment["m00"])
        Y_Coordinate =int(CentreMoment["m01"] / CentreMoment["m00"])
        print('{} {}'.format(X_Coordinate,-Y_Coordinate))
