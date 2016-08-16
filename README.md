# Image Recognition
## Plan:
1. Video is read in from video file frame by frame. 
2. Center of laser location is extracted from frame.
3. Center is appended to list of all locations in the video.
4. Path is simplified by Douglas-Peucker algorithm to reduce pixel jitterings and make the path straighter.
5. Path is transformed into times for each engine to run.
6. Directions are executed.
