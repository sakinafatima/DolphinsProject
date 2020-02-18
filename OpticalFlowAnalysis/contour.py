import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('test3.mp4')

frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
color = np.random.randint(0,255,(100,3))
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

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    # finding contours of each pixel that is moving within the frame
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 3)

    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (1280,720))
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
    # print("change points", changed_points)
    print(f"Len: {len(magnitude)}, type: {type(magnitude)}, shape: {magnitude.shape}");
    # mask[..., 0] = angle * 180 / np.pi
    # # Set image value according to the optical flow magnitude (normalized)
    # mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # # Convert HSV to RGB (BGR) color representation
    # rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    #converting 1-d array back to 2-d array with removed values
    # new_angle_array, new_magnitude_array = [], []
    # for idx, (magnitude_array, angle_array) in enumerate(zip(magnitude, angle)):
    #     _angle_array = [i if i >= 1 and i < 6.26 else 0 for i in angle_array ]
    #     _magnitude_array = [i if i > 0.001 else 0 for i in magnitude_array ]
    #     magnitude[idx] = np.array(_magnitude_array)
    #     angle[idx] = np.array(_angle_array)
    #
    # print("length--------------",magnitude.shape, )

    # converting 2D array into 1D array for plotting histogram
    mag1D = np.ravel(magnitude)
    ang1D=np.ravel(angle)

    # # restricting value of angle between more 1 and 2pi-1 radian
    anglegreaterthanzero=[i for i in ang1D if i>=1]
    anglewithinrange =[i for i in anglegreaterthanzero if i<6.26]

    # restricting value of magnitude greater than 0.36 i.e 1e-3 that is .000000001
    # magnitudewithrange = [i for i in mag1D if i>0.001]

    # newangle= np.reshape(anglewithinrange,(1, len(anglewithinrange)))
    # newmagnitude=np.reshape(magnitudewithrange,(1, len(magnitudewithrange)))
    # print ("shape of the new magnitude 2D array-------",newmagnitude.shape)
    # exit(1)

    # print("NEW ANGLE and magnitude IN 2 d AGAIN-------",newangle, " , ", newmagnitude)


    # print("total pixel points in array------", len(ang1D))
    # print("Array with angles greater than 1------", anglegreaterthanzero)
    # print("Array with angles less than 6------", anglewithinrange)
    # print("size of array whose value is greater than 1------", len(anglegreaterthanzero))
    # print("size of array whose value is less than 6.28------", len(anglewithinrange))
    #  plotting angle
    # def update_line(hl, new_data):
    #     hl.set_xdata(np.append(hl.get_xdata(), new_data))
    #     hl.set_ydata(np.append(hl.get_ydata(), new_data))
    #     # plt.relim()
    #     # plt.autoscale_view()

    # hl, = plt.plot([], [])
    # update_line(hl, ang1D)
    # plt.draw()
    # plt.pause(1)

    plt.hist(anglewithinrange, bins=50)
    plt.show()
    plt.pause(1)
    plt.close()
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
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
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
    # cv2.imshow("Dense optical flow", draw_flow(next_gray, flow))
    cv2.imshow("Dense optical flow", dense_flow)
    # out.write(dense_flow)
    plt.show()
    if cv2.waitKey(40) == 27:
        break
plt.imshow()
cv2.destroyAllWindows()
cap.release()
out.release()
