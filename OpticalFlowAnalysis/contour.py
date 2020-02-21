import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as math

cap = cv2.VideoCapture('test3.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
color = np.random.randint(0, 255, (100, 3))
ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)

# Create mask
mask = np.zeros_like(frame1)
# Sets image saturation to maximum
mask[..., 1] = 255

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    # finding contours of each pixel that is moving within the frame
    _, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (1280, 720))
    out.write(image)
    frame1 = frame2
    ret, frame2 = cap.read()
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                        poly_n=5, poly_sigma=1.1, flags=0)
    # Compute the magnitude and angle of the 2D vectors

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # changed_points, array_of_difference = get_points_changed(magnitude, array_of_difference)
    print("magnitude in 2D array------------shape", magnitude, magnitude.shape)
    print("angle in 2D array", angle)

    new_angle_array, new_magnitude_array = [], []
    # angle = np.array((np.where(((angle < 1) & (angle > 6.26)), 0, angle)))
    # magnitude = np.array((np.where(magnitude < 0.001, 0, magnitude)))
    for idx, (magnitude_array, angle_array) in enumerate(zip(magnitude, angle)):
     _angle_array = [i if i >= 1 and i < 6.26 else 0 for i in angle_array ]
     _magnitude_array = [i if i > 0.001 else 0 for i in magnitude_array ]
     magnitude[idx] = np.array(_magnitude_array)
     angle[idx] = np.array(_angle_array)

    flow[..., 0], flow[..., 1] = magnitude, angle  # updating the flow with new values of angle and magnitude

    print("magnitude in new 2D array of magnitude------------shape", magnitude, magnitude.shape)
    print("angle in 2D new 2D array of angle", angle)


    # converting 2D array into 1D array for plotting histogram
    # mag1D = np.ravel(magnitude)
    # ang1D=np.ravel(angle)

    # restricting value of angle between more 1 and 2pi-1 radian
    # anglegreaterthanzero=[i for i in ang1D if i>=1]
    # anglewithinrange =[i for i in anglegreaterthanzero if i<6.26]

    # restricting value of magnitude greater than 0.36 i.e 1e-3 that is .000000001
    # magnitudewithrange = [i for i in mag1D if i>0.001]

    # print("total pixel points in array------", len(ang1D))
    # print("Array with angles greater than 1------", anglegreaterthanzero)
    # print("Array with angles less than 6------", anglewithinrange)
    # print("size of array whose value is greater than 1------", len(anglegreaterthanzero))
    # print("size of array whose value is less than 6.28------", len(anglewithinrange))
    #  plotting angle
    # plt.hist(anglewithinrange, bins=50)
    # plt.show()
    # plt.pause(1)
    # plt.close()
    # # plotting magnitude
    # plt.hist(magnitudewithrange, bins=50)
    # plt.show()
    # plt.pause(1)
    # plt.close()

    # code for drawing circular histograms
    # bins_number = 50  # the [0, 360) interval will be subdivided into this
    # # number of equal bins
    # bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
    # angles = 2 * np.pi * np.array(j3)
    # n, _, _ = plt.hist(angles, bins)
    # plt.clf()
    # width = 2 * np.pi / bins_number
    # ax = plt.subplot(1, 1, 1, projection='polar')
    # bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
    # for  bar in  bars:
    #
    #     bar.set_alpha(0.5)
    # plt.show()
    # plt.pause(1)
    # plt.close()

    # drawing vectors over the contour points based on the result of magnitude and angle from optical flow
    def draw_flow(img, flow, step=8):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        # uncomment these 2 lines
        # for vector field lines on the whole frame
        # fx, fy = flow[y, x].T
        # lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        magnitude, angle = flow[..., 0].reshape(-1), flow[..., 1].reshape(-1)
        # breaking the pixel lists into an image square
        xs = (np.array(range(w * h)) % w)
        ys = np.array(range(w * h)).__ifloordiv__(w) #returns a//=b.
        zipped = list(zip(xs, ys, magnitude, angle))
        # filtering out the points in the angle and magnitude which were set to 0 previously
        filtered = [(x, y, m, a) for (x, y, m, a) in zipped if
                    (m > 0.4 and a > 0) and (x % step == 0) and (y % step == 0)]

        # creating lines additing angles and magnitude with y and x axis of each pixel point respectively
        lines = [[x, y, x + m, y + a] for (x, y, m, a) in filtered]
        #lines = [[x, y, x + (m*(math.cos(a))), y + (m*(math.sin(a)))] for (x, y, m, a) in filtered]
        lines = np.array(lines).reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    mask[..., 0] = angle * 255 / np.pi
    # Set image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    # cv2.imshow('frame1', rgb)
    dense_flow = cv2.addWeighted(frame1, 1, rgb, 2, 0)
    cv2.imshow("Dense optical flow", draw_flow(next_gray, flow))
    # cv2.imshow("Dense optical flow", dense_flow) #for showing dense flow using color hue and saturation

    if cv2.waitKey(40) == 27:
        break
cv2.destroyAllWindows()
cap.release()
out.release()
