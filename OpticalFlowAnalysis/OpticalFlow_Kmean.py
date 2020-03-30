from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.spatial import distance
import cv2 as cv
import math as maths
import time
from random import randint
import matplotlib.pyplot as plt
from ocr import getMagnification
font = cv.FONT_HERSHEY_SIMPLEX
framecount=1
currentdisplacement=[[0,0],[0,0]]
videoname='third_sampletrim.mp4'
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
fps = cap.get(cv.CAP_PROP_FPS)
fpsperframe=1/fps
print("fps per frame",fpsperframe)
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
    cv.drawContours(frame, contours, -1, (0, 255, 0), 2)

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
    #print("contourpoints after np array------",p0, len(p0))
    if len(p0)>0:
        p1, _st, _err = cv.calcOpticalFlowPyrLK(frame_gray, new_gray, p0, None, **lk_params)
        if framecount>1:
             a = np.array(p1)
             b = np.array(p0)
             print("p1 after np array-------",framecount,a)
             print("current dis after np array-------",framecount,b)
             print("shape of p1 and old dis",a.shape, b.shape)
             ret = a-b
             print("result of diff numpy array", ret)
             # dividing difference by fps:
             velocity = np.divide(a, fpsperframe)
             print("velocity after dividing by fps-----", velocity)
             vel= [maths.sqrt(each[0]**2 + each[1]**2) for each in velocity]
             print("velocity after squarerrot----",vel)
            #see to the status array: _st: contains status of the optical flow displacement, if st==1 means there is change in displacement
             print("fitting kmeans")
             if len(p0)<=2:
              kmeans =KMeans(n_clusters=1, random_state=0).fit(np.array(vel).reshape(-1, 1))
              centroid_old=[[]]
             else:
              kmeans = KMeans(n_clusters=3, random_state=0, max_iter=1000, precompute_distances=True, algorithm='full').fit(np.array(vel).reshape(-1, 1))
              print("k means complete")
             #labels = kmeans.labels_
             labels_=kmeans.labels_
             print("Kmeans Labels------",labels_)

             # centroids_new= kmeans.cluster_centers_
             # #centroid_labels = [centroids[i] for i in labels_]
             # print("centroids labels", kmeans.cluster_centers_)
             # #print("mse-----",kmeans.cluster_centers_ / kmeans.cluster_centers_.ravel())
             # fig, ax = plt.subplots(figsize=(6, 6))
             #
             # plt.scatter(centroids_new[:, 0], centroids_new[:, 1], marker='*', s=300,
             #            c='r', label='centroid')
             #plt.show()

            #calculation of eucladien distance in finding similarity between labels of kmean to keep consistency
            # a= np.array([[470.83255,61.81744],[1167.4661,206.58138],[437.43652,184.53271]])
            # if len(centroids_new>1):
            #     distances = np.empty(centroids_new.shape[0])
            #     for i in range(a.shape[0]):
            #      distances[i] =distance.euclidean(a, centroids_new[i])
            #     print("distance between points",distances[i])

             for i, (f2, f1) in enumerate(zip(p1, p0)): #list comprehension
                    a, b = f2.ravel()
                    c, d = f1.ravel()
                    x, y, w, h = box[i]
                    vx=a/fpsperframe
                    vy=b/fpsperframe
                    vel1= maths.sqrt(vx ** 2 + vy ** 2)
                    label = kmeans.predict(np.array(vel1).reshape(-1,1))
                    if (label==0): #black
                         #vis = cv.circle(vis, (a, b), 2, (0, 0, 0), -1)
                         color=(0, 0, 0)
                         drawRectangle(magnification,color,vis, x, y, w, h)
                    elif (label == 1):
                            # pinks
                            #vis = cv.circle(vis, (a, b), 4, (255, 0, 255), -1)
                            color=(255, 0, 255)
                            drawRectangle(magnification, color, vis, x, y, w, h)
                    elif (label == 2):
                            #purple
                            color = (128, 128, 255)
                            drawRectangle(magnification, color, vis, x, y, w, h)
                    print("-------------Video Name: ", videoname, " --------frame count: ", framecount,
                          " ---------Object Coordinates: ", x, y)
    #cv.imshow('frame1', vis)
    currentdisplacement = p1
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
