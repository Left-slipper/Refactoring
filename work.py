from copy import copy
from pickle import NONE
from scipy.ndimage import gaussian_filter as gs
from PIL import ImageEnhance
import numpy as np
import cv2 as cv
from PIL import Image
from datetime import datetime
import time


path = "binary_photo/"
thresh = 145
                        
            
    
def vector_comparison(dataVectors, img, names, method='ORB'):
    start_time = datetime.now()
    
    img = cv.resize(img, dsize=[265, 325])    
    
    lowe_ratio = 0.8
    key = None
    goodDescript = []
    
    if method   == 'ORB':
        finder = cv.ORB_create()
    elif method == 'SIFT':
        finder = cv.xfeatures2d.SIFT_create()
        
    kp = finder.detect(img,None)
    kp, des = finder.compute(img, kp)
    bf = cv.BFMatcher_create()
    
    name_id = 0
    for vector in dataVectors:
        goodDescript.clear()
        vector = np.uint8(vector)
        matches = bf.knnMatch(des,vector, k=2)

        for m,n in matches:
            if m.distance < lowe_ratio*n.distance:
                goodDescript.append([m])
        
        if(len(goodDescript) >= 65):
            print(datetime.now() - start_time)
            print(len(goodDescript))
            print(names[name_id])
            return True
        name_id += 1
    print(len(goodDescript))

    return False
        


def searchLineObject(img, vectors, names, gistagram, gradients):
    global path, thresh 
    key = 0

    # Обработка изображения, изменение контрастности
    enhancer = ImageEnhance.Contrast(Image.fromarray(img, 'RGB'))
    enhancer_img = np.asarray(enhancer.enhance(1.5))
 
        
    imageGray = cv.cvtColor(enhancer_img, cv.COLOR_BGR2GRAY)  
    imgGray = imageGray.reshape(len(imageGray), len(imageGray[0]))
    imgGray = np.array([imgGray,imgGray,imgGray])
    imgGray = np.moveaxis(imgGray, 0, -1)
    

    gausimg = cv.GaussianBlur(imgGray, (5,5), 0)
    img_binary = cv.threshold(gausimg, thresh, 255, cv.THRESH_BINARY)[1]



    lower = [255, 255, 255]
    upper = [255, 255, 255]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    kernel = np.ones((15,15),np.uint8)
    img_morf = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel)
    mask = cv.inRange(img_morf, lower, upper)
    output = cv.bitwise_and(img_morf, img_morf, mask=mask)
    
    ret,thresh_img = cv.threshold(mask, thresh, 255, 0)
    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    if len(contours) != 0:
        c = max(contours, key = len)

    
        x,y,w,h = cv.boundingRect(c)
        res_img = img[y+25:(y+h-100),x+25:(x+h-120)]
        cv.imshow("result", res_img)
        res_grey_img = imgGray[y+25:(y+h-100),x+25:(x+h-120)]
        res_gaus_img = gausimg[y+25:(y+h-100),x+25:(x+h-120)]
        
        output = copy(img)
        output = cv.drawContours(output, c, -1, 255, 3)
        img_binary = cv.drawContours(img_binary, c, -1, 255, 3)
        cv.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow("cascade_image", output)

    (width, height) = res_img.shape[:2]
    cv.imshow("result_image", img_binary)


    if(vector_comparison(vectors, res_gaus_img, names, 'ORB')):
        key = 0
        cv.destroyAllWindows()

        cv.imshow("result", res_img)
        cv.waitKey(0)  
        

    return key, res_img

cv.waitKey()
cv.destroyAllWindows()

