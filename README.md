# Histogram-of-an-images
## Aim
To obtain a histogram for finding the frequency of pixels in an Image with pixel values ranging from 0 to 255. Also write the code using OpenCV to perform histogram equalization.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Read the gray and color image using imread()

### Step2:
Print the image using imshow().

### Step3:
Use calcHist() function to mark the image in graph frequency for gray and color image.

### step4:
Use calcHist() function to mark the image in graph frequency for gray and color image.

### Step5:
The Histogram of gray scale image and color image is shown.


## Program:
```python
# Developed By: SETHUKKARASI C
# Register Number: 212223230201

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale format.
img = cv2.imread('parrot.jpg', cv2.IMREAD_GRAYSCALE)

# Display the images.
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.show()

assert (img == cv2.imread('parrot.jpg', cv2.IMREAD_GRAYSCALE)).all()

# Display the images
plt.hist(img.ravel(),256,range = [0, 256]);
plt.title('Original Image')
plt.show()

# Equalize histogram
img_eq = cv2.equalizeHist(img)

# Display the images.
plt.hist(img_eq.ravel(), 256, range = [0, 256])
plt.title('Equalized Histogram')

# Display the images.
plt.imshow(img_eq, cmap='gray')
plt.title('Original Image')
plt.show()

# Read the color image.
img = cv2.imread('parrot.jpg', cv2.IMREAD_COLOR)

# Convert to HSV.
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Perform histogram equalization only on the V channel, for value intensity.
img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:, :, 2])

# Convert back to BGR format.
img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

plt.imshow(img_eq[:,:,::-1]); plt.title('Equalized Image');plt.show()

plt.hist(img_eq.ravel(),256,range = [0, 256]); plt.title('Histogram Equalized');plt.show()

# Display the images.
#plt.figure(figsize = (20,10))
plt.subplot(221); plt.imshow(img[:, :, ::-1]); plt.axis("off"); plt.title('Original Color Image')
plt.subplot(222); plt.imshow(img_eq[:, :, ::-1]); plt.axis("off"); plt.title('Equalized Image')
plt.subplot(223); plt.hist(img.ravel(),256,range = [0, 256]); plt.axis("on"); plt.title('Original Image')
plt.subplot(224); plt.hist(img_eq.ravel(),256,range = [0, 256]); plt.axis("on"); plt.title('Histogram Equalized');
plt.show()

# Display the histograms.
plt.figure(figsize = [15,4])
plt.subplot(121); plt.hist(img.ravel(),256,range = [0, 256]); plt.title('Original Image')
plt.subplot(122); plt.hist(img_eq.ravel(),256,range = [0, 256]); plt.title('Histogram Equalized')
```
## Output:
### Input Grayscale Image and Color Image

<img width="552" height="396" alt="image" src="https://github.com/user-attachments/assets/0fe3fd36-074e-4092-b4e9-52c03c06fa10" />

<img width="552" height="396" alt="image" src="https://github.com/user-attachments/assets/d0e2ce91-dd39-443a-88a5-a1a44bbef190" />

### Histogram of Grayscale Image and any channel of Color Image

<img width="1244" height="374" alt="image" src="https://github.com/user-attachments/assets/5b674dc9-30da-4925-adfd-441554f7cdef" />

### Histogram Equalization of Grayscale Image.

<img width="578" height="426" alt="image" src="https://github.com/user-attachments/assets/41fa8121-c91b-4bb4-ae95-6f94d856c08f" />


## Result: 
Thus the histogram for finding the frequency of pixels in an image with pixel values ranging from 0 to 255 is obtained. Also,histogram equalization is done for the gray scale image using OpenCV.
