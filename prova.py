import cv2
import numpy as np
import matplotlib.pyplot as plt


x = np.zeros((500, 600)).astype(np.uint8)
Mx, My = x.shape
c = (int(0), int(My/4))
cv2.ellipse(x, center=c, axes=(200, 200), angle=0, startAngle=0, endAngle=35, color=255, thickness=-1)
cv2.drawMarker(x, position=c, thickness=3, color=250)
print(35)
plt.imshow(x, cmap='gray')
plt.show()

