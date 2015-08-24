# -*- coding: utf-8 -*-
"""Commande moteurs : Commande les 2 moteurs en utilisant des sons générés avec des fréquences différentes"""
from time import sleep
from pygame import mixer

def moteur(freq,temps):
    mixer.init()
    nom=str(freq)+"Hz"
    mixer.music.load("Sons\\{}.wav".format(nom))
    for i in range(temps):
        mixer.music.play()
        sleep(0.1)
    
