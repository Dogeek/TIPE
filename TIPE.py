# -*- coding: Latin-1 -*-
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def GetThresh(image): ######CETTE FONCTION TRANSFORME LA PHOTO EN NOIR ET BLANC
    #Conversion en niveaux de gris et flou
    grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(grey,5)
    #Puis en N&B
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
    #On sauvegarde pour réouvrir avec PIL
    cv2.imwrite("thresh.png", thresh)
    return thresh

def FindContours(thresh): #####CETTE FONCTION TROUVE LES CHIFFRES
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    nb=0
    for cnt in contours:
        approx=cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True) #Approximation du contour
        if len(approx)==4 and cv2.contourArea(cnt)>=40: ######PAS SUR KSA MARCHE########Retravailler cette condition pour ne selectionner que les chiffres les bons contours
            [x,y,w,h] = cv2.boundingRect(cnt)#on associe un rectangle au chiffre
            nb+=1
            IsolateDigits(thresh,nb,x,y,w,h)
    return nb #nb est le nombre de chiffres detectés

def IsolateDigits(img,nb,x,y,w,h):####ISOLE LES CHIFFRES ET LES SAUVEGARDE
    roi = img[x:x+w,y:y+h]
    cv2.imwrite("Digits\\Digit_{}.png".format(nb), roi)

def GetData(nom):####RECUPERE LES DONNEES DE L'IMAGE ENVOYEE EN PARAMETRES ET RENVOIE LA MATRICE ASSOCIEE
    #On ouvre l'image grace à PIL
    img=Image.open("{}".format(nom))
    imgdata=img.getdata()
    larg,haut = img.size
    tab = np.array(imgdata)
    #Mise en forme avec numpy
    matrice=np.reshape(tab, (haut,larg))
    return matrice,haut,larg #On renvoie la matrice de l'image

def CountDigit(nom,nb_verticals = 3,nb_horizontals = 3): #LA FONCTION LA PLUS BOURRIN : COMPTE LE NOMBRE D'INTERSECTIONS ENTRE DES DROITES ET LES CHIFFRES
    matrice,h,w = GetData(nom)
    count_y=[]
    count_x=[]
    for j in range(nb_horizontals):
        count_x[j]=0
    for j in range(nb_verticals):
        count_y[j]=0
    for vert in range(w/nb_verticals):
        for horiz in range(h):
            if matrice[vert][horiz]>128:
                count_y[vert]+=1
    for horiz in range(h/nb_horizontals):
        for vert in range(w):
            if matrice[vert][horiz]>128:
                count_x[horiz]+=1
    return count_x,count_y

def NumberRecognition(): ####FONCTION PRINCIPALE
    #PARAMETRES
    ##############
    Nombre_de_verticales = 5
    Nombre_d_horizontales = 5
    Numero_de_cam=0
    ##############
    #On prend une photo avec la webcam
    image = Snapshot("image",Numero_de_cam)
    #image = cv2.imread("image.png")
    #Noir et Blanc
    thresh = GetThresh(image)
    #On trouve les chiffres
    nb=FindContours(thresh,image)
    #On compte chaque le nombre d'intersections avec une ligne
    count=[]
    model=[]
    solutions=[]
    for i in range(nb):
        nom="Digits\\Digit_{0}.png".format(i)
        x,y=CountDigit(nom,Nombre_de_verticales,Nombre_d_horizontales)
        count[i]=[x,y]
    for j in range(10):
        nom="Models\\Model_{0}.png".format(j)
        x,y=CountDigit(nom,Nombre_de_verticales,Nombre_d_horizontales)
        model[j]=[x,y]
    for k in count:
        for i in range(10):
            if model[i]==k:
                solutions.append(i)
    return solutions

def CodePostal(solutions):#MET LE RESULTAT DE LA RECONNAISSANCE SOUS FORME D'UN NOMBRE UNIQUE
    code_postal=0
    for i in range(len(solutions)):
        code_postal+=solutions[i]*(10**i)
    return code_postal

def Snapshot(nom="image",device=0, extension="png"): #PRENDS UNE PHOTO
    cam=cv2.VideoCapture(device)
    ret, img = cam.read()
    cv2.imwrite("{}.{}".format(nom,extension),img)
    #On désalloue la mémoire
    cam.release()
    return img
