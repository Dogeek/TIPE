# -*- coding: cp1252 -*-
import cv2

def Snapshot(name="image",device=1, extension="png"): #PREND UNE PHOTO
    cam=cv2.VideoCapture(device)
    ret, img = cam.read()
    cv2.imwrite("{}.{}".format(name,extension),img)
    #Memory release
    cam.release()
    return img

Snapshot("test",0, "png")
