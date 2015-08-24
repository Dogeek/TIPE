# -*- coding: cp1252 -*-
import numpy as np
#from VideoCapture import Device
from PIL import Image, ImageOps
import cv2
from time import sleep
#source http://en.wikibooks.org/wiki/Applied_Robotics/Sensors_and_Perception/Open_CV/Basic_OpenCV_Tutorial
reanalyse=False
laplace = False
superposition=False
"""
version 2
"""
def prise():
    if reanalyse == False:#si true reprend une photo
        cam=Device()
        sleep(1)
        nom = "test_auto_v2.jpg".format(numero=0)
        cam=0

    img_orig=cv2.imread("ordi.png".format(numero=0),cv2.CV_LOAD_IMAGE_GRAYSCALE)#charge la dernière image prise en niveau de gris (moins de bug)
    return img_orig

def ANALYSE(img_orig,threshold=170,n=-1):

    if laplace == True:#alors créée image avec laplace
        img_laplace=cv2.Laplacian(img_orig,cv2.CV_16S)
        cv2.imwrite('laplace.jpg', img_laplace)
        print "image laplace créée"
        img_a_analyse1=img_laplace
    else:#sinon on anylse l'image de base
        img_a_analyse1=img_orig

    print img_a_analyse1
    (thresh, img_threshold) = cv2.threshold(img_a_analyse1, threshold, 255, cv2.THRESH_BINARY)#effectue le seuillage
    cv2.imwrite('threshold.jpg', img_threshold)#crée l'image seuillée

    print " image seuillée créée seuil:{}".format(threshold)
    print img_threshold
    (contours, hierarchy) = cv2.findContours(img_threshold, mode=cv2.cv.CV_RETR_CCOMP, method=cv2.cv.CV_CHAIN_APPROX_SIMPLE)#cherche les contours

    if superposition==True:
        results_image=img_orig#on superpose a l'image d'origine
    else:
        results_image = np.zeros((img_threshold.shape[0], img_threshold.shape[1]),np.uint8)#créée une image vide pour y dessiner les contours

#il faut trier les contour par longeur

    if n==-1:
        contours_a_dessiner=contours#de base tous
    else:
        Lcontour=[]
        if n>len(contours):
            n=len(contours)
            print "nombre de seuil max atteint n={}".format(n)
        for i in range(n):
            Lcontour.append(contours[i])
        contours_a_dessiner=np.array(Lcontour)

    cv2.drawContours(results_image,contours_a_dessiner,-1,(255,255,255),2)#dessine les contours
    cv2.imwrite('contour_test_v2.jpg', results_image)#créée l'image
    return

img_orig=prise()
ANALYSE(img_orig,50,100)
'''
#prise

def obtenir_image(numero):
    #sleep(1)
    nom = "test_auto.jpg".format(numero)
    cam.saveSnapshot(nom)
if reanalyse==False:
    cam = Device()
    obtenir_image(numero=0)
    cam=0

#chargement

#http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#imread
img_orig = cv2.imread('test_auto_modifiee.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)#permet de ce passer du passage au gris par la suite


###Laplace https://github.com/abidrahmank/OpenCV2-Python/blob/master/Official_Tutorial_Python_Codes/3_imgproc/laplacian.py

if laplace==True:
    img_laplace=cv2.Laplacian(img_orig,cv2.CV_16S)
    cv2.imwrite('laplace.jpg', img_laplace)
    print 'image laplace crée'
    img_COLOR=img_laplace
else:
    img_COLOR=img_orig

#threshold

##
# Converts an RGB image to grayscale, where each pixel
# now represents the intensity of the original image.
##
def rgb_to_gray(img):
    #return ImageOps.grayscale(img)#en niveau de gris
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

##
# Converts an image into a binary image at the specified threshold.
# All pixels with a value <= threshold become 0, while
# pixels > threshold become 1
def do_threshold(image, threshold =170):
    (thresh, im_bw) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return (thresh, im_bw)



#####################################################
# If you have captured a frame from your camera like in the template program above,
# you can create a bitmap from it as follows:

img_gray=img_orig#image chargé en niveauu de gris
#img_gray = rgb_to_gray(img_COLOR) # Convert img_COLOR from video camera from RGB to Grayscale

# Converts grayscale image to a binary image with a threshold value of 220. Any pixel with an
# intensity of <= 220 will be black, while any pixel with an intensity > 220 will be white:
(thresh, img_threshold) = do_threshold(img_orig, 170)
cv2.imwrite('threshold.jpg', img_threshold)
print 'image threshole crée'

#contour

##
# Finds the outer contours of a binary image and returns a shape-approximation
# of them. Because we are only finding the outer contours, there is no object
# hierarchy returned.
##
def find_contours(image):
    (contours, hierarchy) = cv2.findContours(image, mode=cv2.cv.CV_RETR_EXTERNAL, method=cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    return contours



#####################################################
# If you have created a binary image as above and stored it in "img_threshold"
# the following code will find the contours of your image:
contours = find_contours(img_threshold)
print contours
print (img_threshold.shape[1],img_threshold.shape[0])
print contours[1:3]
# Here, we are creating a new RGB image to display our results on
#bug ori results_image = new_rgb_image(img_threshold.shape[1], img_threshold.shape[0])
#fond noir
results_image = np.zeros((img_threshold.shape[0], img_threshold.shape[1]),np.uint8)

#results_image=img_orig
print results_image
#contours = np.array([[[10,10],[10,20],[20,20],[20,10]]])
#contours = []
#results_image=cv2.drawContours(results_image, contours, -1, cv2.cv.RGB(255,0,0),-1)

cv2.drawContours(results_image,contours,-1,(255,255,255),2)
print results_image

# Display Results
#cv2.imshow("results", results_image)

cv2.imwrite('contour_test.jpg', results_image)
'''
