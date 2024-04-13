#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the video file "ball.mov"
cap = cv2.VideoCapture('ball.mov')

# given ball width
ball_width = 11

# Loop through the video frames
while True:
    # Read a frame
    ret, frame = cap.read()
    
    # Exit if no frames to read
    if not ret:
        break
        
    # Converting the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Applying Gaussian blur to reduce the noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Applying Hough transform (Houghcircles) to detect the ball in the vedio
    ball = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=ball_width*2,
                            param1=100, param2=30, minRadius=5, maxRadius=15)
    
    # Draw the detected circles on the frame (Green)
    if ball is not None:
        ball = np.round(ball[0, :]).astype("int")
        for (x, y, r) in ball:
            cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
            
            # plot the circles on the each frame using Matplotlib
            circle = plt.Circle((x, y), r, color='r', fill=False)
            fig, ax = plt.subplots()
            ax.imshow(frame)
            ax.add_artist(circle)
            plt.show()
    
    # Show the frame
    cv2.imshow("Ball detection", frame)
    
    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the video and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




