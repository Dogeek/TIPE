# -*- coding: utf-8 -*-
import cv2


def Snapshot(nom="image",photo=False,device=0, extension="png"): #PRENDS UNE PHOTO
    if photo:
        cam=cv2.VideoCapture(device)
        ret, img = cam.read()
        cv2.imwrite("{}.{}".format(nom,extension),img)
        #On décharge l'espace mémoire attribué à la caméra spécifiée avec "device"
        cam.release()
    else:
        img=cv2.imread("{}.{}".format(nom,extension))
    return img

def GetThresh(image, Canny=False, cannyparam1=100,cannyparam2=200): ######CETTE FONCTION TRANSFORME LA PHOTO EN NOIR ET BLANC
    if Canny:
        thresh=cv2.Canny(image,cannyparam1,cannyparam2)
    else:
        #Conversion en niveaux de gris et flou
        grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(grey,5)
        #Puis en N&B
        thresh,seuil = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
        print("seuil:{0}".format(seuil)
    #On sauvegarde pour réouvrir avec PIL
    cv2.imwrite("thresh.png", thresh)
    return thresh

def FindContours(thresh, seuil=40): #####CETTE FONCTION TROUVE LES CHIFFRES
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    nb=0
    coord=[]
    for cnt in contours:
        approx=cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True) #Approximation du contour
        if cv2.contourArea(cnt)>= seuil: ######PAS SUR KSA MARCHE########Retravailler cette condition pour ne selectionner que les chiffres les bons contours
            [x,y,w,h] = cv2.boundingRect(cnt)#on associe un rectangle au chiffre
            coord.append([x+w/2,y-h/2])
            nb+=1
    return nb,coord #nb est le nombre de chiffres detectés

seuil=40
photo=False

