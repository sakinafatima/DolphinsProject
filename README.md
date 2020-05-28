# DolphinsProject
The purpose of the project is to develop software that can analyse video images from drones, to locate dolphin schools, count numbers in the schools, and identify the species. Tuna and Dolphins often swim together and during the fishing of Tuna, many dolphins are unintentionally caught. This project aims to help in locating the dolphin schools to improve their conservation and management.
OpticalFlowVelocity.py:
Background Subtraction, global thresholding, contour finding, pixel intensity techniques of opencv are used to find the moving objects in the frame. after that 
optical flow LK method is applied to calculate the delta displacement. From this displacement, velocity is calculated. The objects are then clustered on the basis of their velocity through different Clustering Algorithms.
