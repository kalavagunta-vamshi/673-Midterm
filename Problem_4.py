#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np

# Load the image and resize the window to fit the image
img = cv2.imread("hotairbaloon.jpg")
cv2.namedWindow("Baloon", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Baloon", img.shape[1], img.shape[0])

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#add gaussian blur to reduce the noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to separate the balloons from the background
a, thresh = cv2.threshold(blur, 120, 250, cv2.THRESH_BINARY)

# Find contours of the balloons
contours, a = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Count the number of balloons and draw each balloon with a different color
num_balloons = 0
for contour in contours:
    # Calculate the area of the contour to filter out small noise regions
    area = cv2.contourArea(contour)
    if area > 10000:
        # Draw the balloon with a random color and label it with a number
        # Numpy array is not iterable so I kept in list and draw contours only accept tuple so i converted into tuple
        
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        cv2.drawContours(img, [contour], -1, color, 3)
        cv2.putText(img, str(num_balloons+1), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 3)
        num_balloons += 1

# Print the number of balloons detected
print(f"Number of balloons detected on the image: {num_balloons}")

# Show the image
cv2.imshow("Baloon", img)

# Wait for a key press and then exit
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




