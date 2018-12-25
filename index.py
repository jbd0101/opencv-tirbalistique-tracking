#Import libraries
import numpy as np
import sys,time,math,csv,sys
import cv2,datetime
from tkinter import *
from tkinter import filedialog as fd
#Access video file
print('''
  _______ _____ _____     ____  ____  _      _  ____  _    _ ______
 |__   __|_   _|  __ \   / __ \|  _ \| |    (_)/ __ \| |  | |  ____|
    | |    | | | |__) | | |  | | |_) | |     _| |  | | |  | | |__
    | |    | | |  _  /  | |  | |  _ <| |    | | |  | | |  | |  __|
    | |   _| |_| | \ \  | |__| | |_) | |____| | |__| | |__| | |____
    |_|  |_____|_|  \_\  \____/|____/|______|_|\___\_\\____/|______|

  ''')



video_path = ''
distanceX=233.7
distanceY=123.0
cv2.ocl.setUseOpenCL(False)
version = cv2.__version__.split('.')[0]
print("Version de opencv "+str(version))
window = Tk()
window.geometry("500x400")
LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)
def popupmsg(msg):
  popup = Tk()
  popup.wm_title("!")
  label = Label(popup, text=msg, font=NORM_FONT)
  label.pack(side="top", fill="x", pady=10)
  B1 = Button(popup, text="OK", command = popup.destroy)
  B1.pack()
  popup.mainloop()

print("---------Récupération information------------- ")
try:
  with open('lastusedinputs.csv') as csvfile:
    inputs = csv.reader(csvfile, delimiter=',')
    for row in inputs:
      if(row[0]=="x"):
        distanceX=float(row[1])
      elif(row[0]=="y"):
        distanceY = float(row[1])
      elif(row[0]=="video_path"):
        video_path = str(row[1])
except Exception as e:
  print(e)
  print("no configuration file found")

print("--------- Démarrage questionnement utilisateur ----------")

def selectVideo():
  global window, video_path
  window.filename = fd.askopenfilename()
  video_path = window.filename
  popupmsg("vidéo sélectionné : "+video_path)
def submitData():
  global inputY,inputX,video_path,distanceX,distanceY,window
  if(len(str(inputY.get())) < 1 or len(str(inputX.get())) < 1 or len(video_path)<2):
    popupmsg("Veuillez entrer toutes les valeurs SVP")
  else:
    distanceX = float(inputX.get())
    distanceY = float(inputY.get())
    data = [
    ["x",distanceX],
    ["y",distanceY],
    ["video_path",video_path]
    ]
    outputfile = open('lastusedinputs.csv', 'w')
    with outputfile:
       writer = csv.writer(outputfile)
       writer.writerows(data)

    window.destroy()

b = Button(window, text="Select. VIDEO", command=selectVideo)
b.pack()
if(len(video_path)>2):
  l = Label(window, text="Derniere video utilise: \n "+video_path,font=NORM_FONT)
  l.pack()
l = Label(window, text="distance x (cm avec virg)",font=NORM_FONT)
l.pack()
inputX = Entry(window, bd =5)
inputX.pack()
l = Label(window, text="distance y (cm avec virg)",font=NORM_FONT)
l.pack()
inputY = Entry(window, bd =5)
inputY.pack()
submit = Button(window, text="OK", command=submitData)
submit.pack()

if(len(str(distanceX))>1):
  inputX.insert(END, distanceX)
  inputY.insert(END, distanceY)
window.mainloop()
"""
get data of user input
"""

print(str(distanceX))
print(str(distanceY))

"""
end  user input

"""

# Read video file
cap = cv2.VideoCapture(video_path)

#Set number of frames and create background
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)

fgbg = cv2.createBackgroundSubtractorMOG2(history=100,detectShadows=0,varThreshold = 2000)

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
lastframe = 0
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
      if cv2.contourArea(c) < 1100: #Modify with respect to the moving object size
        continue
      #Get bounding box from contour
      (x, y, w, h) = cv2.boundingRect(c)#gets value for box
      #Draw bounding box (in green)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(frame,str(cv2.contourArea(c)),(x,y), font, 0.3,(255,255,255),1,cv2.LINE_AA)
      # cv2.imshow('rgb',frame)
      for pos in points: #add red dots for past positions
        cv2.circle(frame,(int(pos[0]),int(pos[1])), 2, (0,0,255), 2)
        cv2.putText(frame,str(round(pos[0]*distParPixelX,0))+"cm et "+str(round(distanceY-(pos[1]*distParPixelY),0)),(int(pos[0]+10),int(pos[1])), font, 0.3,(255,255,255),1,cv2.LINE_AA)


      t = nmbrF / fps
      lastframe = nmbrF
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

      #Give points position in pixels
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
      cv2.imshow('res',frame)
      out.write(frame)

  if nmbrF/length*100>=(95):
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
      y.append(float(pos[1]/100.0))
    x = np.array(x)
    y = np.array(y)
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    pv = np.polyder(p)
    print("distance x en fonction du temps")
    print("y="+str(z[0])+"x²+"+str(z[1])+"x"+str(z[0]))
    print("v'=|"+str(abs(z[0]*2))+"x+"+str(z[1])+"|")

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

    break
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break
cap.release()
cv2.destroyAllWindows()




cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_COUNT, lastframe)

# cap.set(cv2.CAP_PROP_POS_MSEC,nmbrF)      # Go to the 1 sec. position
ret,frame = cap.read()                   # Retrieves the frame at the specified second
x = []
y = []
correctedX = []
correctedY = []
for pos in points:
  x.append(int(pos[0]))
  y.append(int(pos[1]))
x = np.array(x)
y = np.array(y)
z = np.polyfit(x, y, 2)
p = np.poly1d(z)

for i in range(0,frame_width,2):
  cv2.circle(frame,(int(i),int(p(i))), 2, (255,255,255), 1)

for pos in points: #add red dots for past positions
  te = abs(int(pos[1]) - int(p(pos[0]))) / int(p(pos[0])) * 100
  if(te > 50):
    cv2.circle(frame,(int(pos[0]),int(pos[1])), 2, (0,0,255), 2)
  else:
    correctedX.append(int(pos[0]))
    correctedY.append(int(pos[1]))
    cv2.circle(frame,(int(pos[0]),int(pos[1])), 2, (0,255,0), 2)

  cv2.putText(frame,str(round(pos[0]*distParPixelX,0))+"cm et "+str(round(distanceY-(pos[1]*distParPixelY),0)),(int(pos[0]+10),int(pos[1])), font, 0.3,(255,255,255),1,cv2.LINE_AA)

x = np.array(correctedX)
y = np.array(correctedY)
z = np.polyfit(x, y, 2)
p = np.poly1d(z)
for i in range(0,frame_width,2):
  cv2.circle(frame,(int(i),int(p(i))), 2, (0,255,0), 2)


cv2.imwrite("result.jpg", frame)          # Saves the frame as an image
cv2.imshow("Frame Name",frame)           # Displays the frame on screen
cv2.waitKey()



data = [["x (m) ","y (m)","t (s)", "x (pixel)","y (pixel)","taux d'erreur"]]
for i in range(len(pointsCM)-1):
  data.append([(pointsCM[i][0]/100.0),(pointsCM[i][1]/100.0),pointsCM[i][2],points[i][0],points[i][1],(abs(int(points[i][1]) - int(p(points[i][0]))) / int(p(points[i][0])) * 100)])
now = datetime.datetime.now()
now = str(now.strftime("%d_%m_%Y-%H:%M"))
outputfile = open('sortie_'+now+'.csv', 'w')
with outputfile:
   writer = csv.writer(outputfile)
   writer.writerows(data)
