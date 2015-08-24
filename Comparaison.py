#-*- coding:utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Simon
#
# Created:     23/11/2014
# Copyright:   (c) Simon 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import cv2
import matplotlib.pyplot as plt
import numpy as np

def ComparaisonMatchShapes(chiffre):
    ret=[]
    img1 = cv2.imread('{}.png'.format(chiffre),0)
    r, thresh = cv2.threshold(img1, 127, 255,0)
    small = cv2.resize(thresh,(100,100))
    contours,hierarchy = cv2.findContours(small,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contourAreas=[]
    for i in range(len(contours)):
        contourAreas.append(cv2.contourArea(contours[i]))
    cnt1 = contours[contourAreas.index(max(contourAreas))]
    for i in range(10):
        img2 = cv2.imread('Models/Model_{}.png'.format(i),0)
        r, thresh2 = cv2.threshold(img2, 127, 255,0)
        small2= cv2.resize(thresh2,(100,100))
        contours,hierarchy = cv2.findContours(small2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt2 = contours[0]
        ret.append(cv2.matchShapes(cnt1,cnt2,1,0.0))
        return ret.index(min(ret))

def ComparaisonPoint(chiffre,resize, contouraprendre=0):
    ret=[]
    liste=[]
    listeproba=[0]*10
    distances=[]
    digit=cv2.imread("{}.png".format(chiffre),0)
    digit=cv2.adaptiveThreshold(digit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
    digit=cv2.resize(digit,(resize,resize))
    (contourDigit,r)=cv2.findContours(digit, mode=cv2.cv.CV_RETR_CCOMP, method=cv2.cv.CV_CHAIN_APPROX_NONE)
    contourDigitEbruite=[]
    for cnt in contourDigit:
        if cv2.contourArea(cnt)>40:
            contourDigitEbruite.append(cnt)
    result=np.zeros((digit.shape[0], digit.shape[1]),np.uint8)
    cv2.drawContours(result,np.array(contourDigitEbruite),-1,(255,255,255),2)
    cv2.imwrite("res.png",result)
    for i in range(10):
        model=cv2.imread("Models/Model_{}.png".format(i),0)
        model=cv2.adaptiveThreshold(model,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
        model=cv2.resize(model,(resize,resize))
        (contourModel,r)=cv2.findContours(model, mode=cv2.cv.CV_RETR_CCOMP, method=cv2.cv.CV_CHAIN_APPROX_NONE)
        #for test in contourDigitEbruite:
        for points in contourDigitEbruite[contouraprendre]:
            (x,y)=points[0][0],points[0][1]
            dist=cv2.pointPolygonTest(contourModel[0],(x,y),True)
            distances.append(dist)
        ret.append(distances)
        distances=[]
    for i in range(len(ret[0])):
        liste=[]
        for j in range(len(ret)):
            liste.append(ret[j][i])
        #mini=min(liste)
        listeproba[liste.index(min(liste))]+=1 #out of range ?
    print listeproba
    return listeproba.index(max(listeproba))

def ComparaisonTemplate(chiffre,resize):
    ret=[]
    #digit=cv2.imread("{}.png".format(chiffre),cv2.IMREAD_GRAYSCALE)
    digit=cv2.imread("CHIFFRES.png",cv2.IMREAD_GRAYSCALE)
    #digit=cv2.resize(digit,(resize,resize))
    for i in range(10):
        template=cv2.imread("Models/Model_{}.png".format(i),cv2.IMREAD_GRAYSCALE)
        #template=cv2.resize(template,(resize,resize))
        res = cv2.matchTemplate(digit,template,cv2.TM_CCOEFF_NORMED)
        plt.subplot(121),plt.imshow(res,cmap='gray')
        plt.subplot(122),plt.imshow(digit,cmap='gray')
        min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
        top_left=max_loc
        if top_left == (0,0):
            ret.append(i)
    return ret
        
    
if __name__ == '__main__':
    chiffre=1
    resize=100
    ComparaisonTemplate(chiffre,resize)
    #print ComparaisonMatchShapes(chiffre)
    #for chiffre in range(10):
        #print ComparaisonPoint(chiffre,resize,0)
        #print ComparaisonMatchShapes(chiffre)
        #print ComparaisonTemplate(chiffre,resize)
