#Import libraries
import numpy as np
import cv2
import sys,time,math

#Access video file
video_path = 'OutChamp.MOV'
cv2.ocl.setUseOpenCL(False)
version = cv2.__version__.split('.')[0]
print("Version de opencv "+str(version))

# Read video file
cap = cv2.VideoCapture(video_path)

#Set number of frames and create background
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)

fgbg = cv2.createBackgroundSubtractorMOG2(history=1000,detectShadows=1,varThreshold = 2000)

"""
Arguments of createBackgroundSubtractorMOG2():
history : states how many previous frames are used for building the background model.
detectShadows : if True, the algorithm will detect shadows and mark them.
varThreshold : threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
"""

# Window dimensions (in pixels)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
t = 0

#Calibration of X and Y axis (in centimeters)
distanceX=233.7
distanceY=123.0
distParPixelX =distanceX / frame_width
distParPixelY =distanceY / frame_height

#Font choice
font = cv2.FONT_HERSHEY_SIMPLEX

#Lists creation
points = []
pointsCM = []
vitessesX = []
vitessesY = []
vitessesC = []
#Output file
out = cv2.VideoWriter('trajectory.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

#Main loop
lastframe = None
nmbrF =0
while (cap.isOpened):
  ret, frame = cap.read()#Reading frame by frame
  if ret==True: #Checking if there is a frame
    nmbrF+=1
    fgmask = fgbg.apply(frame)#Background in black and moving object in white/grey-ish
    (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#Moving object contour

    """
    #Grid implementation (optional)
    for n in range(0,frame_width,int(frame_width/10)):
        cv2.line(frame,(int(n),0),(int(n),int(frame_height)),(255,0,0),2)
    for n in range(0,frame_height,int(frame_height/10)):
        cv2.line(frame,(0,int(n)),(int(frame_width),int(n)),(250,0,0),2)
    """
    for c in contours:
      if cv2.contourArea(c) < 300: #Modify with respect to the moving object size
        continue
      #Get bounding box from contour
      (x, y, w, h) = cv2.boundingRect(c)#gets value for box
      #Draw bounding box (in green)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      # cv2.putText(frame,str(cv2.contourArea(c)),(x,y), font, 0.3,(255,255,255),1,cv2.LINE_AA)
      # cv2.imshow('rgb',frame)
      for pos in points: #add red dots for past positions
        cv2.circle(frame,(int(pos[0]),int(pos[1])), 2, (0,0,255), 2)
        cv2.putText(frame,str(round(pos[0]*distParPixelX,0))+"cm et "+str(round(distanceY-(pos[1]*distParPixelY),0)),(int(pos[0]+10),int(pos[1])), font, 0.3,(255,255,255),1,cv2.LINE_AA)


      t = nmbrF / fps
      lastframe = frame
      pointsCM.append([((x+(w/2.0))*distParPixelX),(distanceY-(y+(h/2.0))),t])
      points.append([(x+(w/2.0)),(y+(h/2.0)),t])
      #Display cm and time
      cv2.putText(frame,str(round((x+(w/2.0))*distParPixelX,2))+"cm et "+str(round(distanceY-((y+(h/2.0))*distParPixelY),2))+"cm",(10,10), font, 0.3,(255,255,255),1,cv2.LINE_AA)
      cv2.putText(frame,str(t)+"s",(frame_width-50,10), font, 0.3,(255,255,255),1,cv2.LINE_AA)
      #Get velocities
      try:
        deltaT = abs(pointsCM[-2][2] - pointsCM[-1][2])
        deltaT = 1 if deltaT == 0 else deltaT

        vitesseX = abs(pointsCM[-2][0] - pointsCM[-1][0]) /deltaT
        vitesseY = abs(pointsCM[-2][1] - pointsCM[-1][1]) /deltaT
        vitesseC = math.sqrt(vitesseX**2 + vitesseY**2)
        #if(vitesseX<12 or vitesseX>20):#fix bugs
        if(1==2):#fix bugs
          cv2.putText(frame,"Erreur",(int(frame_width/2 - 20),60), font, 1,(0,0,255  ),1,cv2.LINE_AA)
        else:
          vitessesX.append([vitesseX,t])
          vitessesY.append([vitesseY,t])
          vitessesC.append([vitesseC,t])
        cv2.putText(frame,"X"+str(round(vitesseX,2))+"cm/s",(int(frame_width/2 - 20),15), font, 0.5,(0,255,0  ),1,cv2.LINE_AA)
        cv2.putText(frame,"Y"+str(round(vitesseY,2))+"cm/s",(int(frame_width/2 - 20),30), font, 0.5,(0,255,0  ),1,cv2.LINE_AA)
        cv2.putText(frame,"C"+str(round(vitesseC,2))+"cm/s",(int(frame_width/2 - 20),45), font, 0.5,(0,255,255 ),1,cv2.LINE_AA)

      except Exception as e:
        pass

      cv2.imshow('res',frame)#show black and white video
      out.write(frame)
  print(str(nmbrF/length*100))
  if nmbrF>=(length-1):

    #Give points position in pixels
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
    print("y="+str(z[0])+"x³+"+str(z[1])+"x²+"+str(z[2])+"x"+str(z[3]))
    # print("v="+str(z[0]*3)+"x²+"+str(z[1]*2)+"x+"+str(z[2]))

    print("distance en fonction du temps")
    x = []
    y = []
    for pos in pointsCM:
      x.append(int(pos[2]))
      y.append(int(pos[1]))
    x = np.array(x)
    y = np.array(y)
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    pv = np.polyder(p)
    print("distance x en fonction du temps")
    print("y="+str(z[0])+"x³+"+str(z[1])+"x²+"+str(z[2])+"x"+str(z[3]))
    print("y'="+str(z[0]*3)+"x²+"+str(z[1]*2)+"x+"+str(z[2]))

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
    time.sleep(10)
    break
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break
cap.release()
cv2.destroyAllWindows()
