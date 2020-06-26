# DolphinsProject
The purpose of the project is to develop software that can analyse video images from drones, to locate dolphin schools, count numbers in the schools, and identify the species. Tuna and Dolphins often swim together and during the fishing of Tuna, many dolphins are unintentionally caught. This project aims to help in locating the dolphin schools to improve their conservation and management.
OpticalFlowVelocity.py:
Background Subtraction, global thresholding, contour finding, pixel intensity techniques of opencv are used to find the moving objects in the frame.  
Optical flow LK method is applied to calculate the delta displacement. From this displacement, velocity is calculated. The objects are then clustered on the basis of their velocity through different Clustering Algorithms (KMeans, HDBScan).
<img width="1022" alt="test04_april20" src="https://user-images.githubusercontent.com/25576435/83128337-31aa2800-a0d3-11ea-8e51-f5418e6b24cc.PNG">
Lastly Methods of Supservised learning like SVM, Random Forest, MLP, Logistic Regression are applied and evaluated to find and predict the dolphins school and to seperate them from the non dolphin objects.
