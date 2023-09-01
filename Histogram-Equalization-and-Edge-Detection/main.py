#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install numpy')


# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


def Sobel_edge_detection(f):
    grad_x = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize = 3)
    grad_y = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize = 3)
    magnitude = abs(grad_x) + abs(grad_y)   # 求影像梯度
    g = np.uint8(np.clip(magnitude, 0, 255))        # 將值截斷在範圍 [0, 255] 
    ret, g = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # 用Threshold函式求邊緣偵測的結果影像，其中套用OTSU演算法

    return g


# In[22]:


# 讀取image
img = cv2.imread('input_image.bmp', 0)  # Read as grayscale (0)

# histogram equalization
equalized_img = cv2.equalizeHist(img)

# Sobel Edge detection
gradient_x = cv2.Sobel(equalized_img, cv2.CV_64F, 1, 0, ksize=3)        # 計算x方向梯度
gradient_y = cv2.Sobel(equalized_img, cv2.CV_64F, 0, 1, ksize=3)        # 計算y方向梯度
edges = cv2.magnitude(gradient_x, gradient_y)       # 儲存至edges中


# 做梯度大小的範圍調整和二值化處理獲得更清晰圖像
OTSU_edges = Sobel_edge_detection(equalized_img)


# Original histogram
plt.figure(figsize=(20, 4))
plt.subplot(1, 3, 1)
plt.hist(img.ravel(), bins=256, color='gray')
plt.title('Original Histogram')

# equalized histogram
plt.subplot(1, 3, 2)
plt.hist(equalized_img.ravel(), bins=256, color='gray')
plt.title('Equalized Histogram')

# processed image after histogram equalization
plt.subplot(1, 3, 3)
plt.imshow(equalized_img, cmap='gray')
plt.title('Equalized Image')

# Detected edges in the image after edge detection
plt.figure(figsize=(6, 6))
plt.imshow(edges, cmap='gray')
plt.title('Detected Edges')

# Detected edges in the image after edge detection with improving
plt.figure(figsize=(6, 6))
plt.imshow(OTSU_edges, cmap='gray')
plt.title('Detected Edges with improving')

# Show the plots
plt.tight_layout()
plt.show()


# In[24]:


get_ipython().system('jupyter nbconvert --to script main.ipynb')

