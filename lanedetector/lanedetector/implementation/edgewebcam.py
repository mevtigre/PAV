import cv2


def canny_webcam():
    "Live capture frames from webcam and show the canny edge image of the captured frames."

    cap = cv2.VideoCapture("C:/Users/fzlfrkkms/Desktop/SEMINAR/lanedetector/LaneDetectionVideos/videos/cut.mp4")

    while True:
        ret, frame = cap.read()
        print(frame.shape[1])  
        if ret == True:
            cropped = cv2.resize(frame,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        else:
            break

        frame_rgb = cv2.GaussianBlur(cropped, (7, 7), 1.41)
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

        edge = cv2.Canny(frame_gray, 25, 75)
        
        cv2.imshow('Original', frame_rgb)
        cv2.imshow('Canny Edge', edge)

        if cv2.waitKey(20) == ord('q'):  # Introduce 20 milisecond delay. press q to exit.
            break

canny_webcam()