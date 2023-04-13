# Implementation-of-Filters

## Aim:

To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:

Anaconda - Python 3.7

## Algorithm:

### Step1

Import the necessary modules.

### Step2

For performing smoothing operation on a image.

1. Average filter
   kernel=np.ones((11,11),np.float32)/121
   image3=cv2.filter2D(image2,-1,kernel)
2. Weighted average filter
   kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
   image3=cv2.filter2D(image2,-1,kernel1)
3. Gaussian Blur
   gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
4. Median filter
   median=cv2.medianBlur(image2,13)

### Step3

For performing sharpening on a image.

1. Laplacian Kernel
   kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
   image3=cv2.filter2D(image2,-1,kernel2)
2. Laplacian Operator
   laplacian=cv2.Laplacian(image2,cv2.CV_64F)

### Step4

Display all the images with their respective filters.

## Program:

### Developed By : SRIJITH R

### Register Number: 212221240054

```python
import cv2
import numpy as np
image = cv2.imread("rin.jpg")
original_image = image
cv2.imshow('original',original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 1. Smoothing Filters

i) Using Averaging Filter

```Python
kernel1 = np.ones((11,11),np.float32)/121
box_filter = cv2.filter2D(original_image,-1,kernel2)
cv2.imshow('box_filter',box_filter)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

ii) Using Weighted Averaging Filter

```Python
kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
weighted_filter = cv2.filter2D(original_image,-1,kernel2)
cv2.imshow('weighted_filter',weighted_filter)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

iii) Using Gaussian Filter

```Python
gaussian_blur = cv2.GaussianBlur(src = original_image, ksize = (11,11), sigmaX=0, sigmaY=0)
cv2.imshow('gaussian_filter',gaussian_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

iv) Using Median Filter

```Python
median = cv2.medianBlur(src=original_image,ksize = 11)
cv2.imshow('median_filter',median)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. Sharpening Filters

i) Using Laplacian Kernal

```Python
kernel3 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
laplacian_kernel = cv2.filter2D(original_image,-1,kernel3)
cv2.imshow('laplacian_kernel',laplacian_kernel)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

ii) Using Laplacian Operator

```Python
laplacian_operator = cv2.Laplacian(original_image,cv2.CV_64F)
cv2.imshow('laplacian_operator',laplacian_operator)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

## OUTPUT:

### Original Image

![output](ss1.png)

### 1. Smoothing Filters

### Using Averaging Filter

![output](ss2.png)

### Using Weighted Averaging Filter

![output](ss3.png)

### Using Gaussian Filter

![output](ss4.png)

### Using Median Filter

![output](ss5.png)

### 2. Sharpening Filters

### Using Laplacian Kernal

![output](ss6.png)

### Using Laplacian Operator

![output](ss7.png)

## Result:

Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
