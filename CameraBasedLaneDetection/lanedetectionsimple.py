## BASIC LANE DETECTION SCRIPT ## 
# @AUTHOR FAZLI FARUK OKUMUS
# @AUTHOR MEVLUDE TIGRE  
## BASIC LANE DETECTION SCRIPT ## 


## IMPORT PART ##
import cv2
import time
import matplotlib.pyplot as plt
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
    vertices = [(190,height-65),((width/2)-60,(height/2)+120),((width/2)+60,(height/2)+120),(width-25,height-65)] 
    vertices = np.array([vertices],np.int32)
    mask = np.zeros_like(edge_image)
    mask=cv2.fillPoly(mask,vertices,255)
    roi_image = cv2.bitwise_and(edge_image,mask)
    if plot == True:
        cv2.imshow("Frame After preprocessing and ROI", roi_image)
    return roi_image

## HOUGH TRANSFORMATION AND RETURN LINES ## 
def processHough(roi_image):
    return cv2.HoughLinesP(roi_image, rho=2, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=35, maxLineGap=5)

## This functions inspired from https://github.com/Arun-purakkatt/medium_repo/blob/main/road_lane_detection%20(1).py
## TAKE THE AVERAGE OF THE LINES, PROVIDE SOLID LINE ## 
def optLine(image, lines):
    neg_slope = []
    pos_slope = []

    if lines is not None:
      for line_pts in lines:
        x1, y1, x2, y2 = line_pts.reshape(4)
        param_space = np.polyfit((x1, x2), (y1, y2), 1)
        slope = param_space[0]
        intercept = param_space[1]
        if slope > 0:
            pos_slope.append((slope, intercept))
        else:
            neg_slope.append((slope, intercept))
    if neg_slope == []: # this part just for the protection of the code, without it would collapse 
        neg_slope=[(0.0001,0.0001)]
    if pos_slope == []:
        pos_slope=[(0.0001,0.0001)]
    # Need to take averages of all
    avg_right = np.average(neg_slope, axis=0)
    avg_left = np.average(pos_slope, axis=0)
    # this part for the detenrmine final lines
    right_line = avg_line_pts(image, avg_right)
    left_line = avg_line_pts(image, avg_left)
    return np.array([left_line, right_line])

## HELPER FUNCTION FOR THE AVERAGE FUNCTION FIND LINE LENGTH ##
def avg_line_pts(image, average_line):
    slp, intercept = average_line
    y_start= image.shape[0]
    #determine the length of the line
    y_final = int(y_start * (3.25/5))
    # find x values
    x_start = int((y_start - intercept) // slp)
    x_final = int((y_final - intercept) // slp)
    return  np.array([[x_start,y_start],[x_final,y_final]])

## OVERLAP LINES ON ORIGINAL IMAGE 
def outVideo(frame,lane):
    blank_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    if (lane is not None):
        for line in lane:
            [x1, y1], [x2, y2] = line
            if (x1 > frame.shape[1] or x2 > frame.shape[1] ):
                continue
            cv2.line(blank_frame, (x1,y1), (x2,y2), (255, 0, 0), thickness=10)

    lane = lane.reshape(4,2)
    lane[[2, 3],:] = lane[[3, 2],:]

    if np.all(np.less(lane,1920)):
        cv2.fillPoly(frame, np.int32([lane]), (0,255, 0))
    return cv2.addWeighted(frame, 1, blank_frame, 0.8, 0.0)


## MAIN CODE ## 
if __name__ == "__main__":
    # please adopt your own filepath...
    cap = cv2.VideoCapture("C:/Users/fzlfrkkms/Desktop/SEMINAR/lanedetector/LaneDetectionVideos/videos/3.mp4") 
    #save = 'C:/Users/fzlfrkkms/Desktop/SEMINAR/lanedetector/LaneDetectionVideos/frames/simple/harder/'
    count=0
    while cap.isOpened():
        ret,frame=cap.read()
        if ret==True:
            #noisereduc_image=processGauss(frame,kernel_size=(7,7),sigmaX=1.41,plot=True)
            start = time.time()
            gray_image=processGray(frame,plot=False)
            edge_image=processEdge(gray_image,plot=True)
            roi_image=processRoi(edge_image,plot=True)
            lines=processHough(roi_image)
            lines_avg=optLine(frame,lines)
            lane_image=outVideo(frame,lines_avg)
            end = time.time()
            seconds = end - start
            fps  = 1 / seconds
            print("Estimated frames per second : {0}".format(fps))
            print ("Time taken : {0} seconds".format(seconds))


            #cv2.imwrite(save+str(count)+".png", lane_image)
            #count=count+1
            cv2.imshow('Lane Detection', lane_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    print("Number of total count : {0}".format(count))
    cap.release()
    cv2.destroyAllWindows()     
