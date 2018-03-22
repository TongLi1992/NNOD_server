import cv2
from PIL import Image
img = cv2.imread('image1.jpg',1)
img2 = Image.fromarray(img, 'RGB')
img2.show()
