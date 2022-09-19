## BASIC LANE DETECTION SCRIPT ## 
# @AUTHOR FAZLI FARUK OKUMUS
# @AUTHOR MEVLUDE TIGRE  
## BASIC LANE DETECTION SCRIPT ## 


## IMPORT PART ##
import matplotlib.pyplot as plt
import cv2
import numpy as np


## NOISE REDUC WITH GAUSS KERNEL ##
def processGauss(frame,kernel_size=(7,7),sigmaX=1,plot=False):
    noisereduc_image = cv2.GaussianBlur(frame, kernel_size,sigmaX)
    if plot== True:
        cv2.imshow("Image after Gauss Blur", noisereduc_image)
    return noisereduc_image

## CONVERT GRAY ## 
def processGray(image,plot=False):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if plot== True:
        cv2.imshow("Image after convertion to Gray scale", gray_image)
    return gray_image

## CANNY EDGE DETECTION ##
def processEdge(gray_image,plot=False):
    edge_image = cv2.Canny(gray_image, 50, 150)
    if plot == True:
        cv2.imshow("After preprocessing", edge_image)
    return edge_image

## DEFINE ROI AND CROP THE IMAGE ##
def processRoi(edge_image,plot=False):
    height = edge_image.shape[0]
    width = edge_image.shape[1]
    vertices = [(0,height),((width/2)-50,(height/2)+60),((width/2)+50,(height/2)+60),(width,height)]
    vertices = np.array([vertices],np.int32)
    mask = np.zeros_like(edge_image)
    mask=cv2.fillPoly(mask,vertices,255)
    roi_image = cv2.bitwise_and(edge_image,mask)
    if plot == True:
        cv2.imshow("Frame After preprocessing and ROI", roi_image)
    return roi_image

## HOUGH TRANSFORMATION AND RETURN LINES ## 
def processHough(roi_image):
    return cv2.HoughLinesP(roi_image, rho=2, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=5)

## TAKE THE AVERAGE OF THE LINES, PROVIDE SOLID ## 
def average(image, lines):
    left = []
    right = []

    if lines is not None:
      for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    if left == []:
        left=[(0.0001,0.0001)]
    if right == []:
        right=[(0.0001,0.0001)]
    #takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    right_line = make_points(image, right_avg)
    left_line = make_points(image, left_avg)
    return np.array([left_line, right_line])

## HELPER FUNCTION FOR THE AVERAGE FUNCTION FIND LINE LENGTH ##
def make_points(image, average):
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (2.75/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return  np.array([[x1,y1],[x2,y2]])




## OVERLAP LINES ON ORIGINAL IMAGE 
def outVideo(frame,lines):
    blank_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            [x1, y1], [x2, y2] = line
            cv2.line(blank_frame, (x1,y1), (x2,y2), (255, 0, 0), thickness=10)

    lines = lines.reshape(4,2)
    lines[[2, 3],:] = lines[[3, 2],:]

    if np.all(np.less(lines,1920)):
        cv2.fillPoly(frame, np.int32([lines]), (0,255, 0))
    return cv2.addWeighted(frame, 0.8, blank_frame, 0.8, 0.0)


## MAIN CODE ## 
if __name__ == "__main__":
    cap = cv2.VideoCapture("C:/Users/fzlfrkkms/Desktop/SEMINAR/LaneDetectionVideos/videos/cut.mp4")
    while cap.isOpened():
        ret,frame=cap.read()
        if ret==True:
            #noisereduc_image=processGauss(frame,kernel_size=(7,7),sigmaX=1.41,plot=True)
            gray_image=processGray(frame,plot=True)
            edge_image=processEdge(gray_image,plot=True)
            roi_image=processRoi(edge_image,plot=True)
            lines=processHough(roi_image)
            lines_avg=average(frame,lines)
            lane_image=outVideo(frame,lines_avg)
            cv2.imshow('Lane Detection', lane_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()     
