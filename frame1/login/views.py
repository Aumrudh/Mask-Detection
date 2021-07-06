from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas
from PIL import Image,ImageTk
import imutils
from cv2 import dnn,cvtColor,resize,putText,rectangle,waitKey,COLOR_BGR2HSV,VideoCapture,COLOR_BGR2RGB,COLOR_RGB2GRAY,FONT_HERSHEY_SIMPLEX,flip,imencode,CAP_DSHOW,COLOR_BGR2GRAY,CAP_FFMPEG,VideoWriter_fourcc,VideoWriter
import numpy as np
from tkinter import Tk, Label,Button,Entry
from math import pow, sqrt
from playsound import playsound
from django.shortcuts import render
from io import BytesIO
from django.http import HttpResponse
from django.http.response import StreamingHttpResponse
from django.core.files.storage import FileSystemStorage
import pyrebase
import logging
import threading
import time
import os,urllib.request
from django.conf import settings
df = pandas.read_csv("E:/proj/Uni/unisys/frame1/login/val2.csv")
d = {'Yes': 1, 'No': 0}
df['Social Distance Violation'] = df['Social Distance Violation'].map(d)
df['Covid'] = df['Covid'].map(d)
d = {'Mask': 0, 'No mask': 1}
df['Mask/No mask'] = df['Mask/No mask'].map(d)
features = ['Age','Mask/No mask','Social Distance Violation']
X = df[features]
y = df['Covid']
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
f=open("test.txt","w")
f.write("0 0 0 0")
f.close()
firebaseConfig = {
    'apiKey': "AIzaSyD8zRTgCWQc8lsblaFrkgeMUt45FASXkXo",
    'authDomain': "login-1688a.firebaseapp.com",
    'databaseURL': "https://login-1688a-default-rtdb.firebaseio.com",
    'projectId': "login-1688a",
    'storageBucket': "login-1688a.appspot.com",
    'messagingSenderId': "943748015576",
    'appId': "1:943748015576:web:d6172f5b03c0d9f55c7ead",
    'measurementId': "G-RCJY67MQ88"
   };
calculateConstant_x = 300
calculateConstant_y = 800

#https://www.kaggle.com/altaga/facemaskdataset    
#https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/#pyi-pyimagesearch-plus-optin-modal
ageNet=dnn.readNet("E:/proj/Uni/age_net.caffemodel","E:/proj/Uni/age_deploy.prototxt")
faceNet = dnn.readNet("E:/proj/Uni/deploy.prototxt", "E:/proj/Uni/res10_300x300_ssd_iter_140000.caffemodel")
faceNetx= dnn.readNet("E:/proj/Uni/opencv_face_detector_uint8.pb","E:/proj/Uni/opencv_face_detector.pbtxt")
caffeNetwork = dnn.readNet("E:/proj/Uni/SSD_MobileNet_prototxt.txt", "E:/proj/Uni/SSD_MobileNet.caffemodel")
maskNet = load_model("E:/proj/Uni/mask_detector.model")
padding=20
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(00-02)', '(04-06)', '(08-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
ctx={}
debug_frame, frame = VideoCapture("E:/test2.webp").read()

firebase=pyrebase.initialize_app(firebaseConfig)
auth=firebase.auth()
count1=0
count2=0
count3=0
count4=0
f=open("c.txt","w")
f.write("1")
f.close()


def hi(request):
    context={'data':0}
    return render(request,'login/login (1).html',context)

fourcc = VideoWriter_fourcc(*'XVID') 
out = VideoWriter('output.avi', fourcc, 20.0, (640, 480)) 
class VideoCamera(object):
    def __init__(self):
        f=open("a.txt","r")
        s=f.read()
        self.video = VideoCapture(s,CAP_FFMPEG)

    def __del__(self):
        self.video.release()

    def get_frame(self):
               debug_frame, frame= self.video.read()
               nomask=0
               maskp=0
               # We are using Motion JPEG, but OpenCV defaults to capture raw images,
               # so we must encode it into JPEG in order to correctly display the
               # video stream.
               highRisk = set()
               position = dict()
               detectionCoordinates = dict()
               frame = imutils.resize(frame, width=800)
               (h, w) = frame.shape[:2]
               #print(h)#450 800
               blob = dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
               faceNet.setInput(blob)
               detections = faceNet.forward()

               faces = []
               locs = []
               preds = []
               for i in range(0, detections.shape[2]):
                   confidence = detections[0, 0, i, 2]
                   if confidence > 0.5:
                       box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                       (startX, startY, endX, endY) = box.astype("int")
                       (startX, startY) = (max(0, startX), max(0, startY))
                       (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                       face = frame[startY:endY, startX:endX]
                       face = cvtColor(face, COLOR_BGR2RGB)
                       face = resize(face, (224, 224))
                       face = img_to_array(face)
                       face = preprocess_input(face)
                       faces.append(face)
                       locs.append((startX, startY, endX, endY))
               if len(faces) > 0:
                   faces = np.array(faces, dtype="float32")
                   preds = maskNet.predict(faces, batch_size=32)
               resultImg=frame.copy()
               frameHeight=resultImg.shape[0]
               frameWidth=resultImg.shape[1]
               blob=dnn.blobFromImage(resultImg, 1.0, (227, 227), [104, 117, 123], True, False)
               faceNetx.setInput(blob)
               detections=faceNetx.forward()
               faceBoxes=[]
               for i in range(detections.shape[2]):
                   confidence=detections[0,0,i,2]
                   if confidence>0.7:
                       x1=int(detections[0,0,i,3]*frameWidth)
                       y1=int(detections[0,0,i,4]*frameHeight)
                       x2=int(detections[0,0,i,5]*frameWidth)
                       y2=int(detections[0,0,i,6]*frameHeight)
                       faceBoxes.append([x1,y1,x2,y2])
                       rectangle(resultImg, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
               flag=0
               kep=-1
               for faceBox in faceBoxes:
                   face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
                   blob=dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                   ageNet.setInput(blob)
                   agePreds=ageNet.forward()
                   age=ageList[agePreds[0].argmax()]
                   kep=int(age[1]+age[2])
                   #print(kep)
                   flag=1
               for (box, pred) in zip(locs, preds):
                   (startX, startY, endX, endY) = box
                   (mask, withoutMask) = pred
                   label = "MASK" if mask > withoutMask else "NO MASK"
                   if label=="NO MASK" and flag:
                       nomask+=1
                       label=label+" AGE"+age[1:-1]
                       f=open("test.txt","r")
                       s=f.read()
                       f.close()
                       a=list(map(int,s.split()))
                       a[0]+=1
                
                       for i in range(0,len(a)):
                                  a[i]=str(a[i])
                       a=' '.join(a)
                       f=open("test.txt","w")
                       f.write(a)
                       f.close()
                   else:
                       maskp+=1

                       f=open("test.txt","r")
                       s=f.read()
                       f.close()
                       a=list(map(int,s.split()))
                       a[1]+=1
                
                       for i in range(0,len(a)):
                                  a[i]=str(a[i])
                       a=' '.join(a)
                       f=open("test.txt","w")
                       f.write(a)
                       f.close()
                   color = (0, 255, 0) if label == "MASK" else (0, 0, 255)
                  
                   label = "{}".format(label)
                   putText(frame, label, (startX, startY - 10),FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                   rectangle(frame, (startX, startY), (endX, endY), color, 2)
                   if not debug_frame:
                       print("Video cannot opened or finished!")
                       break    
               (imageHeight, imageWidth) = frame.shape[:2]
               pDetection = dnn.blobFromImage(resize(frame, (imageWidth, imageHeight)), 0.007843, (imageWidth, imageHeight), 127.5)
               caffeNetwork.setInput(pDetection)
               detections = caffeNetwork.forward()
               for i in range(detections.shape[2]):
                   accuracy = detections[0, 0, i, 2]
                   if accuracy > 0.4:
                       idOfClasses = int(detections[0, 0, i, 1])
                       box = detections[0, 0, i, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                       (startX, startY, endX, endY) = box.astype('int')
                       if idOfClasses == 15.00:


                           detectionCoordinates[i] = (startX, startY, endX, endY)
                           centroid_x = round((startX+endX)/2,4)
                           centroid_y = round((startY+endY)/2,4)
                           bboxHeight = round(endY-startY,4)    
                           distance = (calculateConstant_x * calculateConstant_y) / bboxHeight
                           centroid_x_centimeters = (centroid_x * distance) / calculateConstant_y
                           centroid_y_centimeters = (centroid_y * distance) / calculateConstant_y
                           position[i] = (centroid_x_centimeters, centroid_y_centimeters, distance)
               flag3=0
               for i in position.keys():
                   for j in position.keys():
                       if i < j:
                           distanceOfBboxes = sqrt(pow(position[i][0]-position[j][0],2) + pow(position[i][1]-position[j][1],2) + pow(position[i][2]-position[j][2],2))
                           #print(distanceOfBboxes)
                           if distanceOfBboxes < 100: 
                                highRisk.add(i),highRisk.add(j)
                                flag3=1
                                f=open("test.txt","r")
                                s=f.read()
                                f.close()
                                a=list(map(int,s.split()))
                                a[2]+=1
                
                                for i in range(0,len(a)):
                                  a[i]=str(a[i])
                                a=' '.join(a)
                                f=open("test.txt","w")
                                f.write(a)
                                f.close()
                           elif distanceOfBboxes<300:

                                f=open("test.txt","r")
                                s=f.read()
                                f.close()
                                a=list(map(int,s.split()))
                                a[3]+=1
                
                                for i in range(0,len(a)):
                                  a[i]=str(a[i])
                                a=' '.join(a)
                                f=open("test.txt","w")
                                f.write(a)
                                f.close()
                               
                                                               

               for xx in position.keys():
                   if xx in highRisk:
                       rectangleColor = (0,0,255)
                       (startX, startY, endX, endY) = detectionCoordinates[xx]
                       rectangle(frame, (startX, startY), (endX, endY), rectangleColor, 2)

               hsv = cvtColor(frame, COLOR_BGR2HSV) 
      
               # output the frame 
               out.write(hsv)

               gray = cvtColor(frame, COLOR_BGR2GRAY)
               
               ret, jpeg = imencode('.jpg', frame)

               context=str(nomask)+str(maskp)+str(x)+str(len(highRisk))
               f=open("b.txt","w")
               f.write(context)
               return jpeg.tobytes()

def x(request):
    return render(request, 'login/fir.html')

def index(request):
    f=open("test.txt","w")
    f.write("0 0 0 0")
    f.close()
    global count3
    k="E:/proj/Uni/unisys/frame1"
    if request.method =='POST':
        upload_files=request.FILES['document2']
        fs =FileSystemStorage()
        name=fs.save(upload_files.name,upload_files)
        url=fs.url(name)
        #print(url)
        k+=url
        f=open("a.txt","w")
        f.write(k)
        f.close()
        #print(k)
        #print(upload_files.name)
        #print(upload_files.size)
    f=open("c.txt","r")
    count3=f.read()
    f.close()
    if count3=='a':
      count3='b'
    else:
      count3='a'
    context={'sv':count3}
    return render(request, 'login/home.html',context)


def gen(camera):
    while True:
        frame = camera.get_frame()
        #for x in range(1000):
        #yield (b'\n')

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_message(msg):
    f=open("test.txt","r")
    s=f.read()
    f.close()
    return '{}'.format(s)


def iterator():
    for i in range(1):
          yield gen_message('iteration ' + str(i))


def test_stream(request):
    stream = iterator()
    response = StreamingHttpResponse(stream, status=200, content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response



def video_feed(request):
    a=StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')
    return  a

def add3(request):
    nomask=request.GET['nomask']
    mask=request.GET['mask']
    a=request.GET['a']
    b=request.GET['b']
    age=request.GET['age']
   
    if int(mask)>1:
      mask=1
    else:
      mask=0
    if int(b)>1:
      b=1
    else:
      b=0 
    K=[[mask,b,age]]  
    y_pred = dtree.predict(K)
    print(y_pred)
    if age==-1:
      s="Insufficient Data" 
    elif y_pred[0]==1:
      s="More Possibilities to get Covid-19"
    else:
      s="Not get Covid-19"
    context={'out':s}
    print(s)
    k='login/loading.html'
    return render(request,k,context)
def add2(request): 
  
   name=request.GET['num1']
   password=request.GET['num2']
   print(name,password)
   check=0

   try:
        #user=auth.sign_in_with_email_and_password(name,password)
        check=1
        print("AS")
   except: 
        check=0
        context={'data':1}
        return render(request,'login/login (1).html',context)
        print("DDD")
        pass

   return render(request,'login/frame1.html')
def add(request):

   debug_frame, frame = VideoCapture("E:/test2.webp").read()
   print(debug_frame)
   k="E:/proj/Uni/unisys/frame1"
   if request.method =='POST':
        upload_files=request.FILES['document']
        fs =FileSystemStorage()
        name=fs.save(upload_files.name,upload_files)
        url=fs.url(name)
        print(url)
        k+=url
        print(k)
        print(upload_files.name)
        print(upload_files.size)
   while True:
               nomask=0
               maskp=0
               
               debug_frame, frame = VideoCapture(k).read()
               x = cvtColor(frame, COLOR_BGR2RGB)
               img = Image.fromarray(x, 'RGB')
               img=img.save('E:/proj/Uni/unisys/frame1/login/static/login/images/inp.jpg')
               if not debug_frame:break
               highRisk = set()
               position = dict()
               detectionCoordinates = dict()
               frame = imutils.resize(frame, width=800)
               (h, w) = frame.shape[:2]
               print(h)#450 800
               blob = dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
               faceNet.setInput(blob)
               detections = faceNet.forward()

               faces = []
               locs = []
               preds = []
               for i in range(0, detections.shape[2]):
                   confidence = detections[0, 0, i, 2]
                   if confidence > 0.5:
                       box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                       (startX, startY, endX, endY) = box.astype("int")
                       (startX, startY) = (max(0, startX), max(0, startY))
                       (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                       face = frame[startY:endY, startX:endX]
                       face = cvtColor(face, COLOR_BGR2RGB)
                       face = resize(face, (224, 224))
                       face = img_to_array(face)
                       face = preprocess_input(face)
                       faces.append(face)
                       locs.append((startX, startY, endX, endY))
               if len(faces) > 0:
                   faces = np.array(faces, dtype="float32")
                   preds = maskNet.predict(faces, batch_size=32)
               resultImg=frame.copy()
               frameHeight=resultImg.shape[0]
               frameWidth=resultImg.shape[1]
               blob=dnn.blobFromImage(resultImg, 1.0, (227, 227), [104, 117, 123], True, False)
               faceNetx.setInput(blob)
               detections=faceNetx.forward()
               faceBoxes=[]
               for i in range(detections.shape[2]):
                   confidence=detections[0,0,i,2]
                   if confidence>0.7:
                       x1=int(detections[0,0,i,3]*frameWidth)
                       y1=int(detections[0,0,i,4]*frameHeight)
                       x2=int(detections[0,0,i,5]*frameWidth)
                       y2=int(detections[0,0,i,6]*frameHeight)
                       faceBoxes.append([x1,y1,x2,y2])
                       rectangle(resultImg, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
               flag=0
               kep=-1
               count1=0
               count2=0
               for faceBox in faceBoxes:
                   face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
                   blob=dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                   ageNet.setInput(blob)
                   agePreds=ageNet.forward()
                   age=ageList[agePreds[0].argmax()]
                   kep=int(age[1]+age[2])
                   print(kep)
                   flag=1
               for (box, pred) in zip(locs, preds):
                   (startX, startY, endX, endY) = box
                   (mask, withoutMask) = pred
                   label = "MASK" if mask > withoutMask else "NO MASK"
                   if label=="NO MASK" and flag:
                       count1+=1
                       nomask+=1
                       label=label+" AGE"+age[1:-1]
                   else:
                       count2+=1
                       maskp+=1
                   color = (0, 255, 0) if label == "MASK" else (0, 0, 255)
                   print(label)
                   label = "{}".format(label)
                   putText(frame, label, (startX, startY - 10),FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                   rectangle(frame, (startX, startY), (endX, endY), color, 2)
                   if not debug_frame:
                       print("Video cannot opened or finished!")
                       break    
               (imageHeight, imageWidth) = frame.shape[:2]
               pDetection = dnn.blobFromImage(resize(frame, (imageWidth, imageHeight)), 0.007843, (imageWidth, imageHeight), 127.5)
               caffeNetwork.setInput(pDetection)
               detections = caffeNetwork.forward()
               for i in range(detections.shape[2]):
                   accuracy = detections[0, 0, i, 2]
                   if accuracy > 0.4:
                       idOfClasses = int(detections[0, 0, i, 1])
                       box = detections[0, 0, i, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                       (startX, startY, endX, endY) = box.astype('int')
                       if idOfClasses == 15.00:


                           detectionCoordinates[i] = (startX, startY, endX, endY)
                           centroid_x = round((startX+endX)/2,4)
                           centroid_y = round((startY+endY)/2,4)
                           bboxHeight = round(endY-startY,4)    
                           distance = (calculateConstant_x * calculateConstant_y) / bboxHeight
                           centroid_x_centimeters = (centroid_x * distance) / calculateConstant_y
                           centroid_y_centimeters = (centroid_y * distance) / calculateConstant_y
                           position[i] = (centroid_x_centimeters, centroid_y_centimeters, distance)
               flag3=0
               for i in position.keys():
                   for j in position.keys():
                       if i < j:
                           distanceOfBboxes = sqrt(pow(position[i][0]-position[j][0],2) + pow(position[i][1]-position[j][1],2) + pow(position[i][2]-position[j][2],2))
                           print(distanceOfBboxes)
                           if distanceOfBboxes < 235: 
                               highRisk.add(i),highRisk.add(j)
                               flag3=1
               for xx in position.keys():
                   if xx in highRisk:
                       rectangleColor = (0,0,255)
                       (startX, startY, endX, endY) = detectionCoordinates[xx]
                       rectangle(frame, (startX, startY), (endX, endY), rectangleColor, 2)

             
               
               frame = cvtColor(frame,COLOR_BGR2RGB)
               img = Image.fromarray(frame, 'RGB')
               gray = cvtColor(np.float32(img),COLOR_RGB2GRAY)
               waitkey = waitKey(0)
               
               img=img.save('E:/proj/Uni/unisys/frame1/login/static/login/images/res.jpg')
               x=nomask+maskp-len(highRisk)
               b=len(highRisk)
               mask=maskp
               if int(mask)>1:
                mask=1
               else:
                mask=0
               if int(b)>1:
                b=1
               else:
                b=0 
               K=[[mask,b,kep]]  
               y_pred = dtree.predict(K)
               print(y_pred)
               if kep==-1:
                s="Insufficient Data" 
               elif y_pred[0]==1:
                s="More Possibilities to get Covid-19"
               else:
                s="Not get Covid-19"
               print(s)
               context={'data':1,'Nomask':nomask,'Mask':maskp,'a':x,'b':len(highRisk),'age':kep,'out':s}
               break
   return render(request,'login/result.html',context)
