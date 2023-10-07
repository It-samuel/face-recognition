import cv2
import face_recognition
import numpy as np
import os
import imutils

#initializing path
path = "Models"
images = []
classNames = []
mylist = os.listdir(path)

#reading images from file path
for cls in mylist:
    curntimg = cv2.imread(f'{path}/{cls}')
    images.append(curntimg)
    classNames.append(os.path.splitext(cls)[0])

#Encoding images
def Encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = Encodings(images)
#initializing camera
cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    imgs = cv2.resize(img,(0,0),None,0.33,0.33)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    img = imutils.resize(img, height=450, width=700)

    facesCurFrame = face_recognition.face_locations(imgs)
    encode = face_recognition.face_encodings(imgs, facesCurFrame)

    for encodeFace, faceloc in zip(encode,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*3,x2*3,y2*3,x1 *3
            cv2.rectangle(img,(x1,y1),(x2,y2+9),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-32),(x2,y2+9),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+5,y2+5,),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        if not matches[matchIndex]:

            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1 *4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-32),(x2,y2+9),(0,0,255),cv2.FILLED)
            cv2.putText(img,"UNKNOWN",(x1+6,y2+6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)



    cv2.imshow('FRAME', img)
    cv2.waitKey(1)
