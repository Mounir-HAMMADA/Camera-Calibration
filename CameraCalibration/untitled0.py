"""
Created on Thu Mar 25 16:28:21 2021

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

images = glob.glob('Test\*.jpg')
images2=glob.glob('calibrage\*.jpg')

for fname in images:
    img = cv.imread(fname)
    img2= cv.imread(fname)
    imggris = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    

    # Find the chess board corners
    
    ret, corners = cv.findChessboardCorners(imggris, (m,n),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(imggris,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, (m,n), corners2,ret)
        
        #image avec la detection
        cv.imshow('image avec la detection',img)
        
        #image original
        cv.imshow('image original',img2)
        
        cv.waitKey(1000)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imggris.shape[::-1],None,None)
        h,  w = img.shape[:2]
        newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite('calibresult.png',dst)
        
        #image non deformer
        cv.imshow('image non deformer',dst)

cv.waitKey()

cv.destroyAllWindows()
