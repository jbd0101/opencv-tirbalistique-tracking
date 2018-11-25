import numpy as np
import cv2
import sys,time
from scipy.ndimage import gaussian_filter
video_path = 'tir.mp4'
cv2.ocl.setUseOpenCL(False)

version = cv2.__version__.split('.')[0]
print("Version de opencv "+str(version))
# https://www.youtube.com/watch?v=LWh9I9Z9tW4
#read video file
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fgbg = cv2.createBackgroundSubtractorMOG2(history=15,detectShadows=0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

distanceX=200.0 #(en cm )
distanceY=150.0 #(en cm )

distParPixelX =distanceX / frame_width
distParPixelY =distanceY / frame_height
# frame_width=float(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
# frame_height=float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
font = cv2.FONT_HERSHEY_SIMPLEX
print(frame_height)

lower_red = np.array([50 ,10,15])
upper_red = np.array([255, 255, 255])
points = []
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
lastframe = None
nmbrF =0
while (cap.isOpened):

  #if ret is true than no error with cap.isOpened
  ret, frame = cap.read()
  if ret==True:
    nmbrF+=1
    fgmask = fgbg.apply(frame)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, lower_red, upper_red)
    # res = cv2.bitwise_and(frame,frame, mask= mask)
    (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(frame, contours, -1, (255,255,0), 3)
    for c in contours:
      if cv2.contourArea(c) < 80 or cv2.contourArea(c) > 380:
        continue
      #get bounding box from countour
      (x, y, w, h) = cv2.boundingRect(c)

      #draw bounding box
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(frame,str(cv2.contourArea(c)),(x,y), font, 0.3,(255,255,255),1,cv2.LINE_AA)
      # cv2.imshow('rgb',frame)
      for pos in points:
        cv2.circle(frame,(int(pos[0]),int(pos[1])), 2, (0,0,255), 2)
        cv2.putText(frame,str(round(pos[0]*distParPixelX,2))+"cm et "+str(round(pos[1]*distParPixelY,2)),(int(pos[0]+10),int(pos[1])), font, 0.3,(255,255,255),1,cv2.LINE_AA)


      lastframe = frame
      cv2.imshow('res',frame)
      points.append([(x+(w/2.0)),(y+(h/2.0))])
      time.sleep(0.1)

      out.write(frame)

      if nmbrF>=(length-1):
        x = []
        y = []
        for pos in points:
          x.append(int(pos[0]))
          y.append(int(pos[1]))
        x = np.array(x)
        y = np.array(y)
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        print("y="+str(z[1])+"x^2+"+str(z[0])+"x+"+str(z[2]))
        for i in range(0,frame_width,6):
          cv2.circle(frame,(int(i),int(p(i))), 2, (255,255,255), 2)
        cv2.imshow('res',frame)

  else:
    input("enter to stop")
    break
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break
cap.release()
cv2.destroyAllWindows()
