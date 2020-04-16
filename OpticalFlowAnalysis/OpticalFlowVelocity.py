from __future__ import print_function
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
import math as maths
import time
from ocr import getMagnification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

font = cv.FONT_HERSHEY_COMPLEX_SMALL
videoname='third_sampletrim.mp4'


def drawRectangle(magnification,color,vis,x, y, w, h):
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
framecount=1
frame = frame[130:1030, 0:1990]
print("timestamp of frame--------first frame", cap.get(cv.CAP_PROP_POS_MSEC))
start_frame_number =0
while cap.isOpened():

    start_frame_number = start_frame_number +5
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame_number)

    ret, frame2 = cap.read()
    magnification = getMagnification(frame2)
    print("timestamp of frame--------",cap.get(cv.CAP_PROP_POS_MSEC))
    if frame2 is not None:
        frame2 = frame2[130:1030, 0:1990]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(frame_gray)
        vis = frame2.copy()
        mask=frame2.copy()
        blur = cv.GaussianBlur(fgmask, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        # finding contours of each pixel that is moving within the frame
        _, contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        print("drawing contours------")
        cv.drawContours(frame, contours, -1, (0, 255, 0), 2)
        for i in range(0, len(contours)):
                cnt = contours[i]
                box.append(cv.boundingRect(cnt))
                x, y, w, h = box[i]
                cx = x + (w / 2)
                cy = y + (h / 2)
                contourpoints.append([cx, cy])
        new_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        p0 = np.array(contourpoints, np.float32)
        print(len(p0))
        if len(p0)>1:
            print("finding optical flow-------------")
            p1, _st, _err = cv.calcOpticalFlowPyrLK(frame_gray, new_gray, p0, None, **lk_params)
            p1array = np.array(p1)
            p0array = np.array(p0)
            print("shape of p1 and old dis",p1array, p0array)
            ret = p1array-p0array
            print("result of diff numpy array", ret)
                 # dividing difference by fps:
            velocity = np.divide(ret, fpsperframe)
            print("velocity after dividing by fps-----", velocity)
            vel= [maths.sqrt(each[0]**2+each[1]**2) for each in velocity]
            print("velocity after squarerrot----", vel, "lenght of velocity before--------",len(vel))
            for i in vel[:]:
                print("velocity value----------------------",i)
                if (i<1) or (i>430):
                    vel.remove(i)
            print("velocity value after range values defined----------------------", vel, "length of vel-----",len(vel))
            print(framecount,"frame count now------------")
            if framecount<=6:
             print("fitting kmeans")
             kmeans = KMeans(algorithm="full", n_clusters=2, random_state=0, max_iter=3000).fit(np.array(vel).reshape(-1, 1))
             print("k means complete only for first run---------")
             # labels = kmeans.labels_
             labels_ = kmeans.labels_
             print("Kmeans Labels------", labels_)

            for i, (f2, f1) in enumerate(zip(p1, p0)): #list comprehension
                        a, b = f2.ravel()
                        c, d = f1.ravel()
                        #for printing lines on frame
                        # mask = cv.line(mask, (a, b), (c, d),
                        #                 (255, 255, 255), 2)
                        #
                        # mask = cv.circle(mask, (a, b), 5,
                        #                    (255, 255, 255), -1)
                        x, y, w, h = box[i]
                        vx=(a-c)/fpsperframe
                        vy=(b-d)/fpsperframe
                        print("velocicty before sqaure root",vy)
                        vel1= maths.sqrt(vx**2+vy**2)
                        if (vel1 >0) and (vel1<=430):

                            print("velocity for prediction-------",vel1)
                            label = kmeans.predict(np.array(vel1).reshape(-1,1))
                            if (label==0): #white
                                 #vis = cv.circle(vis, (a, b), 4, (255, 255, 255), -1)
                                 color=(255, 255, 255)
                                 drawRectangle(magnification,color, vis, x,y, w, h)
                                 print("velocicty in label 0", vel1)
                                 #cv.putText(vis, str(vel1), (a, b), font, 1, (255, 255, 255), 1, cv.LINE_AA)
                            elif (label == 1):
                                    # pinks
                                    #vis = cv.circle(vis, (a, b), 4, (255, 0, 255), -1)
                                    color=(255, 0, 255)
                                    drawRectangle(magnification,color, vis, x,y, w, h)
                                    print("velocicty in label 1", vel1)
                                    #cv.putText(vis, str(vel1), (a, b), font, 1, (255, 0, 255), 1, cv.LINE_AA)
                            # elif (label == 2):
                            #         #purple
                            #         color = (128, 128, 255)
                            #         vis = cv.circle(vis, (a, b), 4, (128, 128, 255), -1)
                            #         drawRectangle(color, vis, x,y, w, h)
                            #         print("velocicty in label 2", vel1)
                                    #cv.putText(vis, str(vel1), (a, b), font, 1, (128, 128, 255), 1, cv.LINE_AA)
                            print("-------------Video Name: ", videoname, " --------frame count: ", framecount,
                                  " ---------Object Coordinates: ", a, b)
        print("rendering image")
        framecount = framecount +5;
        cv.imshow("image", vis)
        cv.imshow("frame2", fgmask)
        cv.imshow("frame", frame)
        #cv.imshow("mask", mask)
        frame = frame2
        del contourpoints[:]
        del box[:]
        # pausing for 2 second
        time.sleep(2)
    if cv.waitKey(40) == ord('q'):
     break
cv.destroyAllWindows()
