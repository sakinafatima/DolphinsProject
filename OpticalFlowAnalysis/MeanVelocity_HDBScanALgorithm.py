from __future__ import print_function
import time
import numpy as np
import cv2 as cv
import math as maths
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
import matplotlib.pyplot  as plt
from ocr import getMagnification
from gps import getAltitude
from HDBSCAN import calculate_velocity
font = cv.FONT_HERSHEY_COMPLEX_SMALL
file="TrainingData/datasetwithPathName/trainWithPATHFILEDATASET.csv"
flag=0
TotalVelocity=[]
rowcounter=0
def ClusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]
with open(file, "r", encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        split = line.split(",")
        videoname = split[0]
        framenumber=int(split[1])
        print(videoname,framenumber)
        rowcounter=rowcounter+1
        print("row counter------------------",rowcounter)
        #videoname="D:/ETP trial survey/Jason's Computer/20191120/20191120_3_100m_30km_Whales/錄製_2019_11_20_10_13_54_900.mp4"
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        # performing background subtraction to remove noise
        fgbg = cv.createBackgroundSubtractorMOG2()
        contourpoints = []
        box = []
        sheet = []
        fileIssue=[]
        cap = cv.VideoCapture(videoname)
        fps = cap.get(cv.CAP_PROP_FPS)
        fpsperframe = 1 / fps
        print(fpsperframe)
        setnumber=framenumber-15
        print("first frame first iteration",setnumber)
        cap.set(cv.CAP_PROP_POS_FRAMES,setnumber)
        ret, frame = cap.read()
        #finding magnification and altitude find dolphin's lenght
        magnification = getMagnification(frame)
        alt=getAltitude(videoname, framenumber, gpsdataPath="gpsdata/")
        dolpLength = 1714 * (magnification / alt) + 16.5  # 22.38*magn + 4.05#old
        #converting dolphins lenght to pixels per meter
        #2 meter is estimated lenht of dolphin
        dolpPixelpersecond=dolpLength/2

        print("magnification and altitude values-----",magnification,alt)
        count = 0
        while cap.isOpened():
            # taking a difference of 20 frames within one iteration to see a change in velocity
            setnumber = setnumber + 15
            print("first frame within iteration", setnumber)
            cap.set(cv.CAP_PROP_POS_FRAMES, setnumber)
            ret, frame2 = cap.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(frame_gray)
            vis3 = frame2.copy()
            blur = cv.GaussianBlur(fgmask, (5, 5), 0)
            _, threshglobal = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)

            dilated = cv.dilate(threshglobal, None, iterations=3)
            _, contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
            print("contour points found-------------------------", p0, len(p0))
            count = count + 1
            if len(p0) > 0:
                p1, _st, _err = cv.calcOpticalFlowPyrLK(frame_gray, new_gray, p0, None, **lk_params)
                print("optical flow calculated as I am inside the po>0 loop")
                print("p1 value----------------------",p1)
                p1array = np.array(p1)
                p0array = np.array(p0)
                # print("shape of p1 and old dis",p1array, p0array)
                ret = p1array - p0array
                # print("result of diff numpy array", ret)
                # dividing difference by fps:
                velocity = np.divide(ret, fpsperframe)
                # print("velocity after dividing by fps-----", velocity)
                vel = [maths.sqrt(each[0] ** 2 + each[1] ** 2) for each in velocity]
            elif len(p0)==0:
                print(" No optical flow calculated as no contour was identified")

            if count == 2:

                print("velocity calculated----Lenght: ",len(vel))
                velocityMeterPerSecond = np.divide(vel,dolpPixelpersecond)
                print("velocity Meter per second calculated----Lenght: ", len(velocityMeterPerSecond))
                data = [videoname, framenumber,rowcounter]
                print(data)
                TotalVelocity.extend(velocityMeterPerSecond)
                print("Total Velocity List now------",len(TotalVelocity),  "row number:", rowcounter)
                if rowcounter==13290:
                    from sklearn.preprocessing import MinMaxScaler,StandardScaler,minmax_scale,RobustScaler
                    #clustering in same class if scaling data so avoiding scaling of data here and directly using the dataset on the clustering algo
                    scaler = StandardScaler()
                    TotalVelocity = scaler.fit_transform(np.array(TotalVelocity).reshape(-1,1))

                    print("Total Velocity after scaling------------------------------",TotalVelocity)
                    # db_default = DBSCAN(eps=0.0375, min_samples=3).fit(TotalVelocity)
                    # labels = db_default.labels_
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=1000,cluster_selection_epsilon=0.2,min_samples=5, leaf_size = 100,prediction_data=True).fit(np.array(TotalVelocity).reshape(-1,1))
                    print("labels----------------------------",clusterer.labels_,)
                    from collections import Counter, defaultdict
                    label_frequency= Counter(clusterer.labels_, )
                    print("frequency of labels:----", label_frequency)
                    calculate_velocity(clusterer,label_frequency)
                    exit(0)
                cap.release()
            # cv.imshow("global thresh", threshglobal)
            # cv.imshow("frame", frame)
            # cv.imshow("vis3", vis3)
            frame = frame2
            del contourpoints[:]
            del box[:]
            # cv.waitKey(50)
            #time.sleep(5)
            # if cv.waitKey(40) == ord('q'):
            #    break
    cv.destroyAllWindows()
