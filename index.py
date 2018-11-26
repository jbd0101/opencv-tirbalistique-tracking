import numpy as np
import cv2
import sys,time,math
from scipy.ndimage import gaussian_filter
video_path = 'tir.mp4'
cv2.ocl.setUseOpenCL(False)

version = cv2.__version__.split('.')[0]
print("Version de opencv "+str(version))
# https://www.youtube.com/watch?v=LWh9I9Z9tW4
#read video file
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fgbg = cv2.createBackgroundSubtractorMOG2(history=0,detectShadows=0,varThreshold = 300)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
t = 0

distanceX=200.0 #(en cm )
distanceY=150.0 #(en cm )
pointsCM = []
distParPixelX =distanceX / frame_width
distParPixelY =distanceY / frame_height
print(distParPixelX)
print(distParPixelY)
# frame_width=float(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
# frame_height=float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
font = cv2.FONT_HERSHEY_SIMPLEX
print(frame_height)

lower_red = np.array([50 ,10,15])
upper_red = np.array([255, 255, 255])
points = []
vitessesX = []
vitessesY = []
vitessesC = []
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
    # res = cv2.bitwise_and(frame,fr380ame, mask= mask)
    (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(frame, contours, -1, (255,255,0), 3)
    # for n in range(0,frame_width,int(frame_width/10)):
    #   cv2.line(frame,(int(n),0),(int(n),int(frame_height)),(255,0,0),2)
    # for n in range(0,frame_height,int(frame_height/10)):
    #   cv2.line(frame,(0,int(n)),(int(frame_width),int(n)),(250,0,0),2)

    for c in contours:
      if cv2.contourArea(c) < 5 or cv2.contourArea(c) > 380:
        continue
      #get bounding box from countour
      (x, y, w, h) = cv2.boundingRect(c)

      #draw bounding box
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(frame,str(cv2.contourArea(c)),(x,y), font, 0.3,(255,255,255),1,cv2.LINE_AA)
      # cv2.imshow('rgb',frame)
      for pos in points:
        cv2.circle(frame,(int(pos[0]),int(pos[1])), 2, (0,0,255), 2)
        cv2.putText(frame,str(round(pos[0]*distParPixelX,0))+"cm et "+str(round(distanceY-(pos[1]*distParPixelY),0)),(int(pos[0]+10),int(pos[1])), font, 0.3,(255,255,255),1,cv2.LINE_AA)


      t = nmbrF / fps *10
      lastframe = frame
      pointsCM.append([((x+(w/2.0))*distParPixelX),(distanceY-(y+(h/2.0))),t])
      points.append([(x+(w/2.0)),(y+(h/2.0)),t])
      cv2.putText(frame,str(round((x+(w/2.0))*distParPixelX,2))+"cm et "+str(round(distanceY-((y+(h/2.0))*distParPixelY),2))+"cm",(10,10), font, 0.3,(255,255,255),1,cv2.LINE_AA)
      cv2.putText(frame,str(t)+"s",(frame_width-50,10), font, 0.3,(255,255,255),1,cv2.LINE_AA)
      try:
        deltaT = abs(pointsCM[-2][2] - pointsCM[-1][2])
        deltaT = 1 if deltaT == 0 else deltaT

        vitesseX = abs(pointsCM[-2][0] - pointsCM[-1][0]) /deltaT
        vitesseY = abs(pointsCM[-2][1] - pointsCM[-1][1]) /deltaT
        vitesseC = math.sqrt(vitesseX**2 + vitesseY**2)
        if(vitesseX<12 or vitesseX>20):
          cv2.putText(frame,"Erreur",(int(frame_width/2 - 20),60), font, 1,(0,0,255  ),1,cv2.LINE_AA)
        else:
          vitessesX.append([vitesseX,t])
          vitessesY.append([vitesseY,t])
          vitessesC.append([vitesseC,t])
        cv2.putText(frame,"X"+str(round(vitesseX,2))+"cm/s",(int(frame_width/2 - 20),15), font, 0.5,(0,255,0  ),1,cv2.LINE_AA)
        cv2.putText(frame,"Y"+str(round(vitesseY,2))+"cm/s",(int(frame_width/2 - 20),30), font, 0.5,(0,255,0  ),1,cv2.LINE_AA)
        cv2.putText(frame,"C"+str(round(vitesseC,2))+"cm/s",(int(frame_width/2 - 20),45), font, 0.5,(0,255,255 ),1,cv2.LINE_AA)

        # print(vitesseX)
      except Exception as e:
        pass
      cv2.imshow('res',frame)
      time.sleep(1)


      out.write(frame)

      if nmbrF>=(length-1):
        print("DESSIN point en pixels")
        x = []
        y = []
        for pos in points:
          x.append(int(pos[0]))
          y.append(int(pos[1]))
        x = np.array(x)
        y = np.array(y)
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        pv = np.polyder(p)



        for i in range(0,frame_width,2):
          cv2.circle(frame,(int(i),int(p(i))), 2, (255,255,255), 2)

        print("DESSIN point en cm")
        x = []
        y = []
        for pos in pointsCM:
          x.append(int(pos[0]))
          y.append(int(pos[1]))
        x = np.array(x)
        y = np.array(y)
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        pv = np.polyder(p)
        print("y="+str(z[0])+"x³+"+str(z[1])+"x²+"+str(z[2])+"x"+str(z[3]))
        # print("v="+str(z[0]*3)+"x²+"+str(z[1]*2)+"x+"+str(z[2]))

        print("vitesse x en fonction du temps")
        x = []
        y = []
        for v in vitessesX:
          x.append(float(v[1]))
          y.append(float(v[0]))
        x = np.array(x)
        y = np.array(y)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        print("y="+str(z[0])+"x+"+str(z[1]))
        print("vitesse y en fonction du temps")
        x = []
        y = []
        for v in vitessesY:
          x.append(float(v[1]))
          y.append(float(v[0]))
        x = np.array(x)
        y = np.array(y)
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        print("y="+str(z[0])+"x²+"+str(z[1])+"x+"+str(z[2]))
        print("vitesse cumulés en fonction du temps")
        x = []
        y = []
        for v in vitessesC:
          x.append(float(v[1]))
          y.append(float(v[0]))
        x = np.array(x)
        y = np.array(y)
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        print("y="+str(z[0])+"x²+"+str(z[1])+"x+"+str(z[2]))

        cv2.imshow('res',frame)
  else:
    input("enter to stop")
    break
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break
cap.release()
cv2.destroyAllWindows()
