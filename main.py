import cv2
import numpy as np
import time
# This variable determines if we want to load color range from memory 
# or use the ones defined in the notebook. 
load_from_disk = True


# If true then load color range from memory
if load_from_disk:
    penval = np.load('penval.npy')

video = cv2.VideoCapture(0)
image = cv2.imread("bg.jpg")


while True:
    
    ret, frame = video.read()
    frame = cv2.flip( frame, 1 )
    frame = cv2.resize(frame , (640, 480))
    image = cv2.resize(image , (640 , 480))
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower 
    # ranges from there
    if load_from_disk:
          
            lower_range = penval[0]
            upper_range = penval[1]
            
    # Otherwise define your own custom values for upper and lower range.
    else:             
       lower_range  = np.array([104, 153, 70])
       upper_range = np.array([30, 30, 0])
    
    mask = cv2.inRange(frame, lower_range, upper_range)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    
    f = frame - res

    f = np.where(f == 0, image, f)
    # displaying the window with normal video
    cv2.imshow("video", frame)
    # displaying the window with removed background
    cv2.imshow("mask", f)
    
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()


