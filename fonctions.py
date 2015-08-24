# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        fonctions.py
# Purpose:     Fonctions de traitement d'image pour réaliser un OCR de chiffres
#
# Author:      Simon Bordeyne
#
# Created:     21/11/2014
# Copyright:   (c) Simon Bordeyne 2014
# Licence:     Creative Commons
#-------------------------------------------------------------------------------
#Importation des bibliothèques nécessaires
import cv2                              #OpenCV
from PIL import Image                   #Python Image Library
from time import strftime, sleep        #time
from pygame import mixer                #PyGame
import numpy as np                      #NumPy
import sys                              #system library
#-------------------------------------------------------------------------------

def main():
    CreerLaplace=False                  #Créé l'image laplacienne
    Superposition=False                 #Superpose les contours trouvés et l'image de base
    PrendrePhoto=False                  #Prendre une photo ?
    threshold=170                       #Valeur du seuil pour le seuil manuel
    n=-1                                #n: nombre de contours à sélectionner. -1 pour tous ceux d'aire >= AireContourMini
    Canny=False                         #Utiliser le Canny Edge Detector pour seuiller ou non
    Manuel=False                        #Seuillage manuel.
    Auto=True                           #Seuillage automatique, adaptatif.
    AireContourMini=100                 #Aire minimum du contour pour qu'il soit selectionné. Sert à enlever les impuretées
    hmin = 30                           #Hauteur minimale du rectangle associé.
    wmin = 20                           #Largeur minimale du rectangle associé.

    log("",True)                        #Init du log
    image=Snapshot("Chiffres",PrendrePhoto) #Photo
    analyse(image,AireContourMini,threshold,n,CreerLaplace,Canny,Manuel,Auto,Superposition) #Analyse de l'image
    thresh,nom=GetThresh(image,Canny,Manuel,Auto,threshold) #Obtention du seuillage
    NombreDeContours=IsolateDigits(thresh,nom,AireContourMini,hmin,wmin)  #Isole les chiffres en se basant sur l'aire du contour, la hauteur et la largeur du rectanbgle associé
    pass

def log(message,FirstTime=False):
    message=str(message)
    with open("log.txt","a") as log:
        if FirstTime:
            log.write("\n--------------\n{0}\n--------------\n".format(strftime("%A %d %B %Y %H:%M:%S")))
        else:
            log.write("\n[{}] {}".format(strftime("%H:%M:%S"),message))

def Snapshot(nom="image",photo=False,device=0, extension="png"): #PREND UNE PHOTO
    if photo:
        cam=cv2.VideoCapture(device)
        ret, img = cam.read()
        cv2.imwrite("{}.{}".format(nom,extension),img)
        log("Photo prise.")
        cam.release() #On décharge l'espace mémoire attribué a la caméra specifiée avec "device"
    else:
        img=cv2.imread("{}.{}".format(nom,extension))
        log("Image Lue")
    return img

def GetThresh(image, Canny=False,Manuel=False,Auto=True, threshold=170,cannyparam1=100,cannyparam2=200, ext="png"): #CETTE FONCTION TRANSFORME LA PHOTO EN NOIR ET BLANC
    nom=str()
    if Canny: #Canny Edge Detector est un algorithme de détection de bord
        img=cv2.Canny(image,cannyparam1,cannyparam2)
        log("Image seuillee créée avec Canny")
        nom="canny"
    elif Manuel:
        #Conversion en niveaux de gris et flou
        grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(grey,5)
        (threshold,img) = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY) #Seuil manuel
        log("Image seuillee créée. Seuil : {}".format(threshold))
        nom="threshold manuel"
    elif Auto:
        #Conversion en niveaux de gris et flou
        grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #blur = cv2.medianBlur(grey,5)
        #Puis en N&B
        img= cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
        log("Image seuillee créée. Seuil : automatique")
        nom="threshold auto"
    cv2.imwrite("{}.{}".format(nom,ext), img)
    return img,nom

def IsolateDigits(img,nom,AireContourMini=100,hmin=50,wmin=20):
    contours,h=cv2.findContours(img, mode=cv2.cv.CV_RETR_CCOMP, method=cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    image_chargee=cv2.imread("{}.png".format(nom))
    NombreDeContours=0
    for cnt in contours:
        if cv2.contourArea(cnt)>=AireContourMini:
            [x,y,w,h] = cv2.boundingRect(cnt)#on associe un rectangle au chiffre
            if h>=hmin and w>=wmin and h<=100 and w<=80:
                NombreDeContours+=1
                RegionOfInterest=image_chargee[y:y+h,x:x+w]
                cv2.imwrite("Digits/Digit_{}.png".format(NombreDeContours), RegionOfInterest)
    log("Nombre de Contours : {}".format(NombreDeContours))
    return NombreDeContours

def GetData(nom, ext="png"):#RECUPERE LES DONNEES DE L'IMAGE ENVOYEE EN PARAMETRES ET RENVOIE LA MATRICE ASSOCIEE
    #On ouvre l'image grace à PIL
    img=Image.open("{}.{}".format(nom,ext))
    log("Image Ouverte avec PIL")
    imgdata=img.getdata()
    larg,haut = img.size
    tab = np.array(imgdata)
    #Mise en forme avec numpy
    matrice=np.reshape(tab, (haut,larg))
    log("Matrice\n{}\nHauteur\t{}\nLargeur\t{}".format(matrice,haut,larg))
    return matrice,haut,larg #On renvoie la matrice de l'image, la hauteur et la largeur

def analyse(img_orig,AireContourMini,threshold=170,n=-1,CreerLaplace=True,Canny=False,Manuel=True,Auto=True, Superposition=False):
    if CreerLaplace: #Si on veut l'image laplacienne
        img_laplace=cv2.Laplacian(img_orig,cv2.CV_16S)
        cv2.imwrite('laplace.png', img_laplace)
        log("Image laplacienne créée")
        img_a_analyser=img_laplace
    else: #Sinon on analyse l'image de base
        img_a_analyser=img_orig

    img,name=GetThresh(img_a_analyser,Canny,Manuel,Auto,threshold)
    (contours, hierarchy) = cv2.findContours(img, mode=cv2.cv.CV_RETR_CCOMP, method=cv2.cv.CV_CHAIN_APPROX_SIMPLE) #Cherche les contours

    if Superposition:
        results_image=img_orig #On superpose a l'image d'origine
    else:
        results_image = np.zeros((img.shape[0], img.shape[1]),np.uint8) #Cree une image vide pour y dessiner les contours

    if n==-1:
        Lcontour=[]
        i=0
        for cnt in contours:
            if cv2.contourArea(cnt)>=AireContourMini:
                i+=1
                Lcontour.append(cnt) #n=-1 : tous les contours trouvés d'aire >=40
        contours_a_dessiner=np.array(Lcontour)
        log("Nombre de contours trouvés d'aire supérieure a {}: {}".format(AireContourMini,i))
    else:
        Lcontour=[]
        if n>len(contours):
            n=len(contours)
            log("Nombre de contours maximum atteint : n={}".format(n))
        for i in range(n):
            Lcontour.append(contours[i])
        contours_a_dessiner=np.array(Lcontour)

    cv2.drawContours(results_image,contours_a_dessiner,-1,(255,255,255),2)#dessine les contours
    cv2.imwrite('contours.png', results_image) #créé l'image finale
    log("Image finale créée.")
    pass

def output(freq,temps):
    mixer.init()
    nom=str(freq)+"Hz"
    log("Sortie allumé : \nFréquence : {} Hz\nTemps:{}".format(freq,temps))
    mixer.music.load("Sons/{}.wav".format(nom))
    for i in range(temps):
        mixer.music.play()
        sleep(0.1)
    pass

def recognize(thresh,AireContourMini,hmin,factor=1):
    digits=[]
    #Entrainement de kNN à partir de deux fichiers
    samples = np.loadtxt('samples.data',np.float32)
    responses = np.loadtxt('responses.data',np.float32)
    responses = responses.reshape((responses.size,1))
    model = cv2.KNearest()
    model.train(samples,responses)
    #Application
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt)>AireContourMini:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>hmin:
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.find_nearest(roismall, k = factor)
                string = str(int((results[0][0])))
                digits.append(string)
    cv2.waitKey(0)
    return digits
    pass

def trainingkNN(img_train,AireContourMini,hmin):
    im=img_train.copy()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    
    
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]

    for cnt in contours:
        if cv2.contourArea(cnt)>AireContourMini:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>hmin:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                cv2.imshow('Training... Type the number in red',im)
                key = cv2.waitKey(0)
                if key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1,100))
                    samples = np.append(samples,sample,0)
    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    log("Entrainement effectue")
    np.savetxt('samples.data',samples)
    np.savetxt('responses.data',responses)
    pass

if __name__ == '__main__':
    main()

