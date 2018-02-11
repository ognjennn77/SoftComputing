import cv2
import numpy as np
from pathlib2 import Path
from sklearn import datasets
from skimage.measure import label, regionprops
from scipy import ndimage
from keras.models import load_model
import os #izbacuje neku gresku i ova i naredna linija su da bi je otklonili
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import argparse

#----------------

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border

from skimage.morphology import closing, square
from skimage.color import label2rgb



model = load_model('mdl.h5')

cv2.useOptimized()
def houghTrans(img):
    #cv2.imshow("pocetak",img)
    #cv2.waitKey(0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img,img, mask= mask)

    gray_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    #cv2.imshow("grey",gray_image)
    #cv2.waitKey(0)
    cannyed_image = cv2.Canny(gray_image, 200, 300)
    #cv2.imshow("cannyed_image",cannyed_image)
    #cv2.waitKey(0)

    lines = cv2.HoughLinesP(
        cannyed_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    #print(lines)

    minx = lines[0][0][0]   #inicijalizacija
    miny = lines[0][0][1]
    maxx = lines[0][0][2]
    maxy = lines[0][0][3]

    temp = len(lines)

      

    for i in  range(temp):  
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        if  x1 < minx:
            miny = y1
            minx = x1
        if  x2 > maxx:
            maxx = x2
            maxy = y2

    cv2.line(img, (minx,miny), (maxx, maxy), (0, 255, 0), 1)
    
    return minx,miny,maxx,maxy



def opnV(vNm):
    v = cv2.VideoCapture(cNm)
    if(v.isOpened()):
        r, f = v.read()
        return r, f
    else:
        return 0







#normala
def getN(x,y,v):
    k1 = (v[1]-v[3])/(v[0]-v[2])
    k2 = float(-1/k1)
    n2 = y-k2*x
    return k2,n2

#tacka na normali
def getP(k,n,v):
   # print "pocetak", k ,n
    k1 = (v[1]-v[3])/(v[0]-v[2])
    n1 = v[1] - k1*v[0]
    x = (n1 - n)/(k-k1)
    y = k1*x + n1
   # print "kraj", k1,n1
    #print "res",x,y
    return x, y






nFrame=None     #broj frejmova
cFrame=0  #trenutni frejm
lX=None         #poslednje x
lY=None         #poslednje y

import collections
list = []

def isOnLine(vector,dot1,dot2):
    global lX
    global lY
    global nFrame
    global cFrame
    global list

    #normale na pravu iz tacaka 1 i 2
    k1,n1 = getN(dot1[0],dot1[1],vector)
    k2,n2 = getN(dot2[0],dot2[1],vector)

    x1,y1 = getP(k1,n1,vector)
    x2,y2 = getP(k2,n2,vector)

    #print "prva ", x1,y1," tacka",dot1[0],dot1[1]
    #print "druga ", x2,y2," tacka",dot2[0],dot2[1]
    #jednacina prave kroz dvije tacke    
    k = float(vector[1]-1-vector[3]-1)/float(vector[0]-1-vector[2]-1)
   #k = float(vector[3]-vector[1])/float(vector[2]-vector[0])
    #n = float(vector[1]-k*vector[0])
    n = float(vector[1]-1 - k*(vector[0]-1))

    #x0 = (dot1[0]+dot2[0])/2
    #y0 = (dot1[1]+dot2[1])/2

  

    # Create a black image
    img = np.zeros((240,320,3), np.uint8)

    
    j=0
    for i in list:
        #print i
        nFrame = i[2]
        lX = i[0]
        lY = i[1]
        if(nFrame!=None):
            if(abs(cFrame-nFrame)<=20):
                if(abs(lX-dot2[0])<=15 and abs(lY-dot2[1])<=15):
                    return False
        j += 1
        if(j==10): break

    if(vector[0]<vector[2]):
        if((vector[0]<dot1[0] and vector[2]>dot1[0]) or (vector[0]<dot2[0] and vector[2]>dot2[0])):
            if(abs(dot2[1] - abs(int((k*dot2[0]) + n)))<3):
                lX = dot2[0]
                lY = dot2[1]
                nFrame=cFrame
                list.insert(0,(dot2[0],dot2[1],cFrame))
                return True
            else:
                return False
    else:
        if((vector[2]<x1 and vector[0]>x1) or (vector[2]<x2 and vector[0]>x2)):
            if(abs(dot2[1] - abs(int((k*dot2[0]) + n)))<30):
                return True
            else:
                return False



def sVideo():
    global cFrame
    global nFrame
    global list
    sumList=[]
    video = "./data/video-"
    for x in range(0, 10):
        sum = 0
        print "{0}{1}{2}".format(video, str(x), ".avi")
        vNm = "{0}{1}{2}".format(video, str(x), ".avi")
        vid = cv2.VideoCapture(vNm)
        vid.set(3,320)
        vid.set(4,240)
        vid.set(5,10)
        #print vid.get(cv2.CAP_PROP_FRAME_COUNT)
        nFrame = None
        cFrame = 0
        list = []
        r, frm = vid.read()
        if(r):
            print "video readed"
        #print "--------------------------", ret
        while(vid.isOpened()):
            cFrame += 1


            x1, y1 , x2, y2 = houghTrans(frm)




            #izdvajanje samo brojeva(crno-bijela slika sa brojevima)
            hsv = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)
            

            #sensitivity = 75
            lower_white = np.array([0,0,180])
            upper_white = np.array([255,75,255])

            mask = cv2.inRange(hsv, lower_white, upper_white)
            res = cv2.bitwise_and(frm,frm, mask= mask)
            #cv2.imshow('res',res)
            #cv2.waitKey(0)

            mPic = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            mPic = cv2.GaussianBlur(mPic, (5, 5), 0)

            imgedged = cv2.Canny(mPic, 50, 100)
            imgedged = cv2.dilate(imgedged, None, iterations=1)
            imgedged = cv2.erode(imgedged, None, iterations=1)
            
            #izdvajanje oblasti tj samo brojeva
            im2, cnts, hierarchy = cv2.findContours(imgedged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rects = [cv2.boundingRect(ctr) for ctr in cnts]

            for rect in rects:
                if(rect[3]>10):
                    #leng = int(rect[3] * 1.1)
                    leng = int(rect[3])
                    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
                    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
                    roi = imgedged[pt1:pt1+leng, pt2:pt2+leng]

                    dot1 = [rect[0],rect[1]]
                    dot2 = [rect[0]+rect[2],rect[1]+rect[3]]
                    onVector = isOnLine((x1,y1,x2,y2),dot1,dot2)
                    
                    

                    #slanje slike na detekciju
                    preparedImg = np.zeros((28, 28))
                    h,w = roi.shape
                    #print h, w
                    o = (28-h)/2
                    for i in range(0, h):
                        for j in range(0, w):
                            if(o+i>=28 or o+j>=28):
                                o=o-1
                            preparedImg[i+o, j+o] = roi[i, j]
                    #cv2.imshow("im for predict",preparedImg)
                    #cv2.waitKey(0)

                    
                    p = model.predict(preparedImg.reshape(1, 784), verbose=1)
                    v = np.argmax(p)
                    # print value
                    if(onVector):
                        sum = sum + v
                    
                    cv2.destroyAllWindows()
            r, frm = vid.read()
            if(r!=True):
                sumList.append((vNm,sum))
                break
    createFile(sumList)


def createFile(sumList):
    with open('./data/out.txt', 'w') as out:
        out.write('RA 180/2014 Ognjen Babalj\n')
        out.write('file	sum\n')
        for sum in sumList:
            s = sum[0].split("/")
            print s
            out.write("{0}{1}{2}{3}".format(s[2], "\t", sum[1],"\n"))
        out.close()


sVideo()
