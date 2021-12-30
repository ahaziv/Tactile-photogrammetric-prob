# Tactile-photogrammetric-probe
This code implements a measurement tool to map 3D coordinates in space using a set of markers 
of which locations are known, and another pointer, of which its’ dimensions are known as well. 
For this task we built the pointer shown in Picture3.
This pointer consists of one marker which is used to identify the pointer in the image and four 
other circles which are used to measure different points around it. The circle points were 
located asymmetrically around the marker to avoid flipping of the different circles around the 
pointers’ axis. The pointer includes 4 points to allow identification of the homography between 
the marker and the reference plane.
It's a fairly simple script, just pull and run.
