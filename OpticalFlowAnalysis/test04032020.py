from __future__ import print_function
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
import time
from random import randint
import matplotlib.pyplot as plt
from ocr import getMagnification
font = cv.FONT_HERSHEY_SIMPLEX
framecount=1
videoname='test3.mp4'
def drawRectangle(magnification,color,vis,x, y, w, h):
    print("magnification-----------------------------------",magnification)
    if magnification <= 4.0:
        if (w < 80 and h < 80):
            vis = cv.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            print("drawing small rectnagles", magnification)
    elif (magnification > 4.0):
                vis = cv.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                print("drawing large rectnagles", magnification)
    return vis

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
fgbg = cv.createBackgroundSubtractorMOG2()
contourpoints=[]
box=[]
cap = cv.VideoCapture(videoname)
ret, frame = cap.read()
frame = frame[130:1030, 0:1990]
while cap.isOpened():
    ret, frame2 = cap.read()
    magnification = getMagnification(frame2)
    frame2 = frame2[130:1030, 0:1990]
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame_gray)
    vis = frame2.copy()
    blur = cv.GaussianBlur(fgmask, (5, 5), 0)
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)
    # finding contours of each pixel that is moving within the frame
    _, contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # for i in range(0, len(contours)):
    #     cnt = contours[i]
    #     cnt = cnt.astype(np.float32)
    for i in range(0, len(contours)):
            cnt = contours[i]
            #cnt = cnt.astype(np.float32)
            box.append(cv.boundingRect(cnt))
            x, y, w, h = box[i]
            cx = x + (w / 2)
            cy = y + (h / 2)
            contourpoints.append([cx, cy])
            # contourpoints.append([x + w, y + h])
            # contourpoints.append([x + w, y])
            # contourpoints.append([x, y + h])
           # print("contourpoints in loop------", contourpoints)
    new_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    #print("contourpoints",contourpoints)
    p0 = np.array(contourpoints, np.float32)
    print("contourpoints after np array------",p0, len(p0))
    if len(p0)>0:
        p1, _st, _err = cv.calcOpticalFlowPyrLK(frame_gray, new_gray, p0, None, **lk_params)
        #print ("result of optical flow",p1)
        #see to the status array: _st: contains status of the optical flow displacement, if st==1 means there is change in displacement
        print("fitting kmeans")
        if len(p0)<=2:
            #random_state makes result reproduceable trying in making k-mean optimezed. as results of kmean is stochastic
         kmeans =KMeans(n_clusters=1, random_state=0).fit(p1)
        else:
         kmeans = KMeans(n_clusters=3, random_state=0).fit(p1)
        print("k means complete")
        #labels = kmeans.labels_
        print("Kmeans Labels------",kmeans.labels_)
        for i, (f2, f1) in enumerate(zip(p1, p0)): #list comprehension
                a, b = f2.ravel()
                c, d = f1.ravel()
                x, y, w, h = box[i]
                label = kmeans.predict([[a, b]])
                if (label==0): #black
                     #vis = cv.circle(vis, (a, b), 2, (0, 0, 0), -1)
                     color=(0, 0, 0)
                     drawRectangle(magnification,color,vis, x, y, w, h)
                elif (label == 1):
                        # pink
                        #vis = cv.circle(vis, (a, b), 4, (255, 0, 255), -1)
                        color=(255, 0, 255)
                        drawRectangle(magnification, color, vis, x, y, w, h)
                elif (label == 2):
                        #purple
                        color = (128, 128, 255)
                        drawRectangle(magnification, color, vis, x, y, w, h)
        print("-------------Video Name: ", videoname, " --------frame count: ", framecount, " ---------Object Coordinates: ", x, y)
    #cv.imshow('frame1', vis)
    print("rendering image")
    framecount=framecount+1;
    cv.imshow("image",vis)
    cv.imshow("frame2", fgmask)
    cv.imshow("frame", frame)
    frame=frame2
    del contourpoints[:]
    del box[:]
    if cv.waitKey(40) == ord('q'):
        break
    time.sleep(2)
cv.destroyAllWindows()
#pausing for 2 seconds
