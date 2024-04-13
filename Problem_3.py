#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('train_track.jpg')

# Resize the image
new_img = cv.resize(image, (800, 600))

# Convert the resized image to grayscale
gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)

# Apply Canny edge detection and Hough line transform
Track_line = cv.HoughLines(cv.Canny(gray, 840, 850, apertureSize=3), 1, np.pi/180, 200)

# Calcuate the ending point
end_point = np.zeros(2)

for line in Track_line:
    r, t = line[0]
    p, q = np.cos(t), np.sin(t)
    x_dot, y_dot = p * r, q * r
        
    # Calculate the coordinates of two endpoints of the line
    x, y = int(x_dot + 1000*(-q)), int(y_dot + 1000*(p))
        
    # Draw the line on the image
    cv.line(new_img, (x, y), (int(x_dot), int(y_dot)), (0, 0, 255), 2)
        
    # Accumulate the coordinates of all the Track_line
    end_point += np.array([x, y])

# Calculate the average ending point
end_point //= (len(Track_line) * 2)

# Calculate transformation matrix
h, w = new_img.shape[:2]
M = cv.getPerspectiveTransform(np.float32([end_point, [end_point[0], h], [w, h], [w, end_point[1]]]), np.float32([[0, 0], [0, h], [w, h], [w, 0]]))

# Apply transformation matrix to get top-down view
Tview = cv.warpPerspective(new_img, M, (w, h))

# Convert top view to grayscale
gray = cv.cvtColor(Tview, cv.COLOR_BGR2GRAY)

# Apply threshold to create binary image
_, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Create mask of train tracks
mask = np.zeros_like(binary)
mask[binary == 0] = 255

# Find distance between train tracks for every row
d = [np.sum(mask[row, :] == 255) for row in range(mask.shape[0])]

# Compute average distance between train tracks
average_distance = np.mean(d)

# Show result
print(f"Average distance in pixels is: {average_distance}")
cv.imshow('Tview', Tview)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:




