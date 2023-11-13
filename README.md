### Vision Aided Navigation project ### 

This repository contains the implementation of my final project for the 'SLAM - Video Navigation' course at HUJI. 

The objective of the project is to build a vision-based navigating system, such that would be able to evaluate the relative movement of the test car along the trajectory.  

### System Input ### 
***
As input, the system gets stereo images, provided by the KITTI project. The input images were taken using the KITTI project car, which is equipped with a stereo camera sensor.

Single frame example: 

![image](https://github.com/dortal721/VANProject/assets/129318571/8ab5f585-70cc-46e8-b096-472a111d8188)

### Code Structure ### 
***
Along the semester, match detection was done mostly using SIFT algorithm (implemented in OpenCV). However, for the final submission SIFT was replaced by a more sophisticated, Deep-Learning based, match detector called ALIKE. The ALIKE detector yielded high-performance matches while maintaining a resonable runtime compared to other deep-learning based models. The ALIKE implementation can be found here: [https://github.com/Shiaoming/ALIKE](url)

final_project.py - This is the main file that runs the entire process. 

VanCollections.py - This file contains implementations of some data structures required to store information throughout the process. 

CNN_detector.py - This file contains an implementation of a class designed to wrap the original implementation of ALIKE. 

performance analysis - This script produces different visualizations required for performance analysis. 

functional module - contains implementations of various computations required for the tracking process. 

ex_code module - contains implementations of some assignments given in the course along the semester. Only necessary for some parts of the performance analysis. 

Note: This repository does not contain the implementation for ALIKE. Hence, in order for the code in here to function, one should obtain it from the creator's github (referenced above).


### Performance ### 
*** 
Tracking video:

https://github.com/dortal721/VANProject/assets/129318571/362f200e-5ca5-470b-9981-8e3241af9f9b  

Point cloud comparison: 

![compare_3d_maps](https://github.com/dortal721/VANProject/assets/129318571/37105d83-5722-4fdd-9e73-da855cf26f87)

Inliers comparison: 

![compare_inliers_rate](https://github.com/dortal721/VANProject/assets/129318571/22c8507e-ca6f-4e0c-bf48-5dd859887f31)

Loop Closure inliers comparison: 

![compare_lc_inliers](https://github.com/dortal721/VANProject/assets/129318571/360fdcef-f8ba-4a41-a015-8c8debd46b8b) 



