# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 10:16:36 2015

@author: Simon
"""
import cv2
import matplotlib.pyplot as plt

resize=100
ret=[]
#digit=cv2.imread("{}.png".format(chiffre),cv2.IMREAD_GRAYSCALE)
digit=cv2.imread("Mode_1.png",cv2.IMREAD_GRAYSCALE)
#digit=cv2.resize(digit,(resize,resize))
template=cv2.imread("Features/feat_1.png".format(2),cv2.IMREAD_GRAYSCALE)
#template=cv2.resize(template,(resize,resize))
res = cv2.matchTemplate(digit,template,cv2.TM_SQDIFF)
min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
mid=min_loc
w, h = template.shape[::-1]
top_left=(mid[0]-w/2,mid[1]-h/2)
bot_right=(mid[0]+w/2,mid[1]+h/2)
cv2.rectangle(res,top_left,bot_right,255,2)
cv2.circle(res,mid,100,255)
#plt.subplot(122),plt.imshow(res,cmap='gray')
#♣plt.subplot(121),plt.imshow(digit,cmap='gray')
#plt.subplot(122),plt.imshow(template,cmap='gray')
plt.imshow(res, cmap='gray')
plt.show()
print top_left
