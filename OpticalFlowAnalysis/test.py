from __future__ import print_function
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
from random import randint
import matplotlib.pyplot as plt
colmap = {1: 'r', 2: 'g', 3: 'b'}

import time

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

fgbg = cv.createBackgroundSubtractorMOG2()
frame_count=0
track_len = 0.1
detect_interval = 5
tracks = []
contourpoints=[]

cap = cv.VideoCapture('trim2.mp4')
#
# def update_tracks(p0, p0r, tracks, p1):
#
#     d = abs(p0 - p0r).reshape(-1, 2).max(-1)
#     good = d < 1
#     new_tracks = []
#
#     for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
#         # Zips tracks the points and whether its true or not into a list/array? dunno
#         # unpacks into tr(integer values of co-ordinates)/ (x,y) floating values and good flag which is boolean
#         # checks if point is good or not if it is then it appends the floating value to the integer
#         if not good_flag:
#             continue
#         tr.append((x, y))
#         # drawcircles
#         cv.circle(vis, (x, y), 1, (0, 255, 0), -1)
#
#         new_tracks.append(tr)
#
#     tracks = new_tracks
#
#     cv.polylines(vis, [np.int32(tr) for tr in tracks], 0, (0, 255, 0))
#     return tracks
#
# def  update_and_drawcontour(contours,boxpoints):
#     for i in range(0, len(contours)):
#         cnt = contours[i]
#         cnt = cnt.astype(np.float32)
#
#         if (len(cnt) > 9):
#             x, y, w, h = cv.boundingRect(cnt)
#             cx = x + (w / 2)
#             cy = y + (h / 2)
#             boxpoints.append([x, y])
#             boxpoints.append([x + w, y + h])
#             boxpoints.append([x + w, y])
#             boxpoints.append([x, y + h])
#             boxpoints.append([cx, cy])
#     for contour in contours:
#      cv.drawContours(vis, contours, -1, (0, 255, 0), 2)
#     boxpoints = np.asarray(boxpoints).astype(np.float32)
#     return boxpoints


def operationsOncontour(contours):
 newframe=frame.copy()
 for i in contours:
    cnt = i
    print ("bounding rect------", cv.boundingRect(cnt))
    x, y, w, h = cv.boundingRect(cnt)

    cv.rectangle(newframe, (x, y), (x + w, y + h), (0, 255, 0), 2)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(newframe, "a", (x, y), font, 4, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow("hull", newframe)

    pass
ret, frame = cap.read()

lin = np.zeros_like(frame)
t = 0
while True:
    t = t + 1
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame_gray)
    vis = frame.copy()
    blur = cv.GaussianBlur(fgmask, (5, 5), 0)
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)
    # finding contours of each pixel that is moving within the frame
    contours, hierarchy = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(vis, contours, -1, (0, 255, 0), 2)

    for i in range(0, len(contours)):
        cnt = contours[i]
        cnt = cnt.astype(np.float32)
        for i in range(0, len(contours)):
            cnt = contours[i]
            cnt = cnt.astype(np.float32)
            x, y, w, h = cv.boundingRect(cnt)
            cx = x + (w / 2)
            cy = y + (h / 2)
            contourpoints.append([x, y])
            contourpoints.append([x + w, y + h])
            contourpoints.append([x + w, y])
            contourpoints.append([x, y + h])
            contourpoints.append([cx, cy])
    ret, previousFrame = cap.read()
    new_gray = cv.cvtColor(previousFrame, cv.COLOR_BGR2GRAY)

    p0 = np.array(contourpoints, np.float32)
    p1, _st, _err = cv.calcOpticalFlowPyrLK(frame_gray, new_gray, p0, None, **lk_params)
    #see to the status array: if status[i] == 1 then new_pts[i] contains a new coordinates of the old_pts[i].
    # good_new = p1[_st == 1]
    # good_old = p0[_st == 1]
    difference=abs(p0 - p1)

    print("fitting kmeans")
    kmeans =KMeans(n_clusters=4, random_state=0).fit(p1)
    print("k means complete")
    #labels = kmeans.labels_
    print("Kmeans Labels------",kmeans.labels_)
    for i, (f2, f1) in enumerate(zip(p1, p0)): #list comprehension
        a, b = f2.ravel()
        c, d = f1.ravel()

        #####
        label = kmeans.predict([[a, b]])
        #label = randint(0,3)
        if (label == 0):
            vis = cv.circle(vis, (a, b), 4, (0, 0, 255), -1)
            #vis = cv.circle(vis, (c, d), 4, (255, 0, 0), -1)
            #lin = cv.line(lin, (a, b), (c, d), (0, 255, 0), 1)
        elif (label == 1):
            vis = cv.circle(vis, (a, b), 4, (255, 0, 255), -1)#pink
        elif (label == 2):
            vis = cv.circle(vis, (a, b), 4, (0, 255, 255), -1)
        elif (label == 3):
            vis = cv.circle(vis, (a, b), 4, (0, 125, 0), -1)

        #img = cv.add(vis, vis)
        ###

        #vis = cv.circle(vis, (a, b), 4, (0, 0, 255), -1)
        #vis = cv.circle(vis, (c, d), 4, (255, 0, 0), -1)
        lin = cv.line(lin, (a, b), (c, d), (0, 255, 0), 1)

    img = cv.add(vis, lin)
    print("p1 and p0r", p0)
    frame=previousFrame
    p0 = np.array(contourpoints, np.float32)
    #cv.imshow('frame1', vis)
    print("rendering image")
    cv.imshow("image",img)
    #cv.imshow("frame2", fgmask)
    #cv.imshow("frame", frame)

    if cv.waitKey(40) == ord('q'):
        break

cv.destroyAllWindows()