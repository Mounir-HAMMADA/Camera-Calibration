# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:30:10 2021

@author: hamad
"""

import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#n et m represente la dimension du chessboard
n=9
m=7
objp = np.zeros((n*m,3), np.float32)
#on prend une chessboard de dimension 6*7
objp[:,:2] = np.mgrid[0:m,0:n].T.reshape(-1,2)
#obtenir a la fin une matrice avec les cordonnee (0,0,0)... (1,1,0), (1,2,0)...(6,5,0)....(m,n,0)

objpoints = [] # 3d point en monde reel
imgpoints = [] # 2d points en plan d image.
capture =cv.VideoCapture('CC-r2.mp4')
while True :
    isTrue, img= capture.read()
    if isTrue == 0:
        break
    imggris = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    #cvtColor pour rendre l'image N&B
    
    ret, corners = cv.findChessboardCorners(imggris, (m,n),None)
    #findchessboardcorners une fonction avec 2 sortie une pour booleen si les corners se trouve une autre la matrice avec les coordonnee initial des coins

    #print(corners)
    
    if ret == True:
        objpoints.append(objp)
        #print(objpoints)
        #print(objp)
        corners2 = cv.cornerSubPix(imggris,corners,(11,11),(-1,-1),criteria)
        #cornersubpix c'est pour trouver l'emplacement precis des coins
        imgpoints.append(corners2)
        img = cv.drawChessboardCorners(img, (m,n), corners2,ret)
        cv.imshow('img',img)
        cv.waitKey(10)
    if ret == False:
        cv.imshow('img',img)
        cv.waitKey(10)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()
 