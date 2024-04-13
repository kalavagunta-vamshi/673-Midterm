#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#problem 2
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


#Problem 3
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


#Problem 4
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

