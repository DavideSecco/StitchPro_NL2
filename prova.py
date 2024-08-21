import cv2
import numpy as np
import matplotlib.pyplot as plt


x = np.zeros((1000, 1000)).astype(np.uint8)
Mx, My = x.shape
c = (int(0), int(720))
value = [0, 720, 360.0, 360.0, -3.141592653589793, 1.5707963267948966]
cv2.ellipse((0, 720), (360.0, 360.0), -3.141592653589793, 1.5707963267948966, 255)
cv2.drawMarker(x, position=c, thickness=3, color=250)
print(35)
plt.imshow(x, cmap='gray')
plt.show()

