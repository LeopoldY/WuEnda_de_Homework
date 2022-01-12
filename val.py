import cv2
import numpy as np

from model.Model import predict

params = np.load(r"output/params.npz")
w = params['arr_0']
b = params['arr_1']

img_org = cv2.imread(r'valPicture/cat_val.jpeg')
img = cv2.resize(img_org, (64,64))
img = img.reshape(64*64*3,1) / 255

if predict(w, b, img).item() == 0:
    print("It's not a cat!")
else:
    print("It's a cat!")

cv2.imshow('1',img_org)
cv2.waitKey()
