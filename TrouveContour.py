#-*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        Contours trouvés
# Purpose:
#
# Author:      Simon
#
# Created:     22/11/2014
# Copyright:   (c) Simon 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import cv2
import numpy as np

def main(hmin,hmax,wmin,wmax,MinArea,b):
    n=0
    if b:
        fichier=input("fichier?")
    else:
        fichier="6.png"
    im = cv2.imread("C:/TIPE/chiffres.jpg")
    im = cv2.resize(im,(640,480))
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    cv2.imwrite("thresh.png",thresh)
    t=cv2.imread("thresh.png")
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt)>MinArea:
            [x,y,w,h] = cv2.boundingRect(cnt)

            if  h>hmin and w>wmin and h<hmax and w<wmax:
                n+=1
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                roi = t[y:y+h,x:x+w]
                cv2.imwrite("roi_{}.png".format(n),roi)
                roismall = cv2.resize(roi,(10,10))
                cv2.imshow('Contours trouves',im)
                cv2.waitKey(0)
    pass

if __name__ == '__main__':
    hmin=63
    hmax=100
    wmin=20
    wmax=80
    MinArea=100
    b=False
    main(hmin,hmax,wmin,wmax,MinArea,b)
    cv2.destroyAllWindows()
