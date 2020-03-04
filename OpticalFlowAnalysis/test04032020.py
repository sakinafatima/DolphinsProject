from __future__ import print_function
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
from random import randint
import matplotlib.pyplot as plt

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

fgbg = cv.createBackgroundSubtractorMOG2()
contourpoints=[]

cap = cv.VideoCapture('zoom_out.mp4')
ret, frame = cap.read()
frame = frame[130:1030, 0:1990]

lin = np.zeros_like(frame)

while cap.isOpened():
    ret, frame2 = cap.read()
    frame2 = frame2[130:1030, 0:1990]
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame_gray)
    vis = frame.copy()
    blur = cv.GaussianBlur(fgmask, (5, 5), 0)
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)
    # finding contours of each pixel that is moving within the frame
    _, contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
    new_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    p0 = np.array(contourpoints, np.float32)
    p1, _st, _err = cv.calcOpticalFlowPyrLK(frame_gray, new_gray, p0, None, **lk_params)
    #see to the status array: _st: contains status of the optical flow displacement, if st==1 means there is change in displacement
    print("fitting kmeans")
    kmeans =KMeans(n_clusters=3, random_state=0).fit(p1)
    print("k means complete")
    #labels = kmeans.labels_
    print("Kmeans Labels------",kmeans.labels_)
    for i, (f2, f1) in enumerate(zip(p1, p0)): #list comprehension
            a, b = f2.ravel()
            c, d = f1.ravel()
            label = kmeans.predict([[a, b]])
            if (label == 0):
                #yellow
                vis = cv.circle(vis, (a, b), 4, (246, 255, 0), -1)
            elif (label == 1):
                # pink
                vis = cv.circle(vis, (a, b), 4, (255, 0, 255), -1)
            elif (label == 2):
                #light blue
                vis = cv.circle(vis, (a, b), 4, (0, 255, 255), -1)
            # elif (label == 3):
            #     vis = cv.circle(vis, (a, b), 4, (0, 125, 0), -1)
            #vis = cv.circle(vis, (a, b), 4, (0, 0, 255), -1)
            lin = cv.circle(lin, (c, d), 4, (255, 0, 0), -1)
            # lin = cv.line(lin, (a, b), (c, d), (0, 255, 0), 1)
    img = cv.add(vis,lin)
    print("p1 and p0r", p0)
    #cv.imshow('frame1', vis)
    print("rendering image")
    cv.imshow("image",img)
    cv.imshow("frame2", fgmask)
    cv.imshow("frame", frame)
    frame=frame2
    if cv.waitKey(40) == ord('q'):
        break
cv.destroyAllWindows()
