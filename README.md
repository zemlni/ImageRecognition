# Image Recognition
## Overview
The robot operates as follows:
1. 30 frames per second are recorded by the RaspberryPi camera. 
2. Using the OpenCV library, the images are processed and the location of the laser is identified.
3. The location is converted from the frame of reference of the camera (as seen on the image) to the frame of the robot (2d as if viewing cartesian plane from above).
4. Directions are sent to each side (two motors on each side) for how long to run.
## Files
The file LaserFunctions.py contains all relevant functions for the robot. Other files are included only for testing purposes and were used in preliminary stages of development.
## Video
This is the implementation of the video seen at https://www.youtube.com/watch?v=TZP6kDM82Fc
