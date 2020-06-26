from __future__ import print_function
import time
import numpy as np
import cv2 as cv
import math as maths
import pandas as pd

from ocr import getMagnification #Dr lewis McMillan Code
from gps import getAltitude #Dr lewis McMillan Code

def calculate_velocity(kmeans,label_frequency):
    file = "LabelsdatabyCREEMTEAM.csv"
    rowcounter = 0
    with open(file, "r", encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            split = line.split(",")
            videoname = split[0]
            framenumber=int(split[1])
            x0=int(split[2])
            y0=int(split[3])
            x1=int(split[4])
            y1=int(split[5])
            labels=split[6]
            print(videoname,framenumber,x0,y0,x1,y1)
            rowcounter=rowcounter+1
            print("row counter------------------",rowcounter)
            lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
            # performing background subtraction to remove noise
            fgbg = cv.createBackgroundSubtractorMOG2()
            contourpoints = []
            box = []
            sheet = []
            cap = cv.VideoCapture(videoname)
            fps = cap.get(cv.CAP_PROP_FPS)
            fpsperframe = 1 / fps
            print(fpsperframe)
            setnumber=framenumber-15
            print("first frame first iteration",setnumber)
            cap.set(cv.CAP_PROP_POS_FRAMES,setnumber)
            difference, frame = cap.read()
            #finding magnification and altitude
            magnification = getMagnification(frame)
            alt=getAltitude(videoname, framenumber, gpsdataPath="gpsdata/")
            dolpLength = 1714 * (magnification / alt) + 16.5
            #converting dolphins lenght to pixels per meter
            #2 meter is estimated lenht of dolphin
            dolpPixelpersecond=dolpLength/2

            #croping frame to given region of interest
            frame = frame[x0:x1,y0:y1]
            count = 0
            while cap.isOpened():
                # taking a difference of 15 frames within one iteration to see a change in velocity
                setnumber = setnumber + 15
                print("first frame within iteration", setnumber)
                cap.set(cv.CAP_PROP_POS_FRAMES, setnumber)
                difference, frame2 = cap.read()
                frame2 = frame2[x0:x1, y0:y1]
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                fgmask = fgbg.apply(frame_gray)
                vis3 = frame2.copy()
                # applying gausian blur and dilation
                blur = cv.GaussianBlur(fgmask, (5, 5), 0)
                _, threshglobal = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)

                dilated = cv.dilate(threshglobal, None, iterations=3)
                _, contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # finding contours
                for i in range(0, len(contours)):
                    cnt = contours[i]
                    box.append(cv.boundingRect(cnt))
                    x, y, w, h = box[i]
                    cx = x + (w / 2)
                    cy = y + (h / 2)
                    contourpoints.append([cx, cy])
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                new_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
                p0 = np.array(contourpoints, np.float32)
                count = count + 1
                if len(p0) > 0:
                    p1, _st, _err = cv.calcOpticalFlowPyrLK(frame_gray, new_gray, p0, None, **lk_params)
                    p1array = np.array(p1)
                    p0array = np.array(p0)

                    difference = p1array - p0array
                    # dividing difference by fps:
                    velocity = np.divide(difference, fpsperframe)

                    velocity_ = [maths.sqrt(each[0] ** 2 + each[1] ** 2) for each in velocity]
                    cv.rectangle(vis3, (x0, y0), (x1, y1), (0, 255, 0), 2)
                elif len(p0)==0:
                    #no motion detected in region of interest so no contour point and hence zero velocity
                    velocity_=0
                if count == 2:
                    #taking mean velocity of all detected contour in region of interest
                    meanVelocity = np.mean(velocity_)
                    #converting velocity to meters per second
                    # ms ^ -1 = (pixels s ^ -1) / (dolphLength / 2)
                    velocityMeterPerSecond = meanVelocity / dolpPixelpersecond

                    #assigning clusters to each region of interest on basis on mean velocity
                    Kmean_label = kmeans.predict(np.array(velocityMeterPerSecond).reshape(-1, 1))
                    if (Kmean_label==0):
                         x=0
                    elif (Kmean_label == 1):
                        x=1
                    elif (Kmean_label == 2):
                        x = 2
                    if (Kmean_label==3):
                         x=3
                    elif (Kmean_label == 4):
                        x=4
                    elif (Kmean_label == 5):
                        x=5
                    data = [videoname, framenumber, x0, y0, x1, y1, labels, meanVelocity,velocityMeterPerSecond,x,rowcounter]
                    print(data)
                    sheet.append(data)
                    df = pd.DataFrame(sheet)
                    df.to_csv('KMeansResult.csv', mode='a', header=False, index=None, encoding='utf_8_sig')
                    cap.release()
                cv.imshow("global thresh", threshglobal)
                cv.imshow("frame", frame)
                cv.imshow("vis3", vis3)
                frame = frame2
                del contourpoints[:]
                del box[:]
                cv.waitKey(50)
        cv.destroyAllWindows()