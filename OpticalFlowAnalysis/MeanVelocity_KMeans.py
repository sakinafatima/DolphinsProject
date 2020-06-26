from __future__ import print_function
import time
import numpy as np
import cv2 as cv
import math as maths
from sklearn.cluster import KMeans
from ocr import getMagnification #Dr lewis McMillan Code
from gps import getAltitude #Dr lewis McMillan Code
from Kmeans import calculate_velocity
font = cv.FONT_HERSHEY_COMPLEX_SMALL
file="TrainingData/dataforUnSupervisedLearning/trainWithPATHFILEDATASET.csv"
flag=0
TotalVelocity=[]
rowcounter=0
def ClusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]
#reading training dataset for unsupervised learning
with open(file, "r", encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        split = line.split(",")
        videoname = split[0]
        framenumber=int(split[1])
        print(videoname,framenumber)
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
        fileIssue=[]
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

        print("magnification and altitude values-----",magnification,alt)
        count = 0
        while cap.isOpened():
            # taking a difference of 15 frames within one iteration to see a change in velocity
            setnumber = setnumber + 15
            print("first frame within iteration", setnumber)
            cap.set(cv.CAP_PROP_POS_FRAMES, setnumber)
            difference, frame2 = cap.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(frame_gray)
            vis3 = frame2.copy()
            # applying gausian blur and dilation
            blur = cv.GaussianBlur(fgmask, (5, 5), 0)
            _, threshglobal = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
            dilated = cv.dilate(threshglobal, None, iterations=3)
            #finding contours
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
            #there should be one contour point detected for optical flow function to work
            if len(p0) > 0:
                p1, _st, _err = cv.calcOpticalFlowPyrLK(frame_gray, new_gray, p0, None, **lk_params)
                print("optical flow calculated as I am inside the po>0 loop")
                print("p1 value----------------------",p1)
                p1array = np.array(p1)
                p0array = np.array(p0)
                difference = p1array - p0array

                # dividing difference by fps:
                velocity = np.divide(difference, fpsperframe)
                velocity_ = [maths.sqrt(each[0] ** 2 + each[1] ** 2) for each in velocity]
            elif len(p0)==0:
                print(" No optical flow calculated as no contour was identified")

            if count == 2:

                #converting pixels per second to meters per second
                velocityMeterPerSecond = np.divide(velocity_, dolpPixelpersecond)
                print("velocity Meter per second calculated----Lenght: ", len(velocityMeterPerSecond))
                data = [videoname, framenumber,rowcounter]
                print(data)
                TotalVelocity.extend(velocityMeterPerSecond)
                print("Total Velocity List now------",len(TotalVelocity),  "row number:", rowcounter)
                if rowcounter==13290:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    TotalVelocity = scaler.fit_transform(np.array(TotalVelocity).reshape(-1,1))

                    # Finding the number of clusters 'K' with elbow method
                    # Using inertia attribute to calculate the sum of squared distances of samples to the nearest cluster centre (elbow method)
                    # squaredDistances = []
                    # K = range(1, 15)
                    # for k in K:
                    #     km = KMeans(n_clusters=k)
                    #     km = km.fit(np.array(TotalVelocity).reshape(-1,1))
                    #     squaredDistances.append(km.inertia_)
                    # print("K Values:", K)
                    # plt.plot(K, squaredDistances, 'bx-')
                    # plt.xlabel('k')
                    # plt.ylabel('Squared Distances')
                    # plt.title('Optimal k')
                    # plt.show()
                    kmeans = KMeans(n_clusters=6, random_state=0, max_iter=300,precompute_distances=True, algorithm='full').fit(np.array(TotalVelocity).reshape(-1,1))
                    print("Kmeans done for the velocities of all points of all frames available in the dataset, row counter :",rowcounter)
                    labels_ = kmeans.labels_
                    print("Kmeans Labels------ are generated", labels_, len(labels_))
                    centroids = kmeans.cluster_centers_

                    labels = kmeans.labels_
                    from collections import Counter, defaultdict
                    label_frequency=Counter(labels)
                    print("frequency of labels:----",label_frequency)
                    #training is done, now velocity of each region of interest will be detected
                    calculate_velocity(kmeans,label_frequency)
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
