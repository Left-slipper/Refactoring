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

def detected_gradient(img, gradients, names):
    return False
    
    img = cv.resize(img, dsize=[10,10])

    for gradCount in range(len(gradients)):        
        resultGrad = 0 
        count = 0
        
        for i in range(10):
            for j in range(9):

                gradImg = [int(img[i][j][0]) - int(img[i][j+1][0]), int(img[i][j][1]) - int(img[i][j+1][1]), int(img[i][j][2]) - int(img[i][j+1][2])] 
                
                #print(sum(gradImg))
                #print(sum(gradients[gradCount][count]))
                print("Grad = " , abs(sum(gradients[gradCount][count]) - sum(gradImg)))
                print("Atalon = ", abs(sum(gradients[gradCount][count]))*0.4)
                
                if(abs(sum(gradients[gradCount][count]) - sum(gradImg)) <= abs(sum(gradients[gradCount][count]))*0.4):
                    resultGrad += 1
                count += 1
                
        print("resultGrad = ", resultGrad)
    
        if(resultGrad >= 15):
            print( names[gradCount])
            time.sleep(3)  # import time
            return True
        else: return False
        
      

def detected_gistagram(img, gistagrams, names):
    return False
    
    img = cv.resize(img, dsize=[20,20])
    h, w = img.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[img[i][j]] += 1
    
   
    name_result = ""     
    for gist, name in zip(gistagrams, names):
        resultGist = 0   
        
        if (len(grayHist) < len(gist)):
            grayHist = np.interp(gist, grayHist, grayHist)
        elif (len(gist) < len(grayHist)):
            gist = np.interp(grayHist, gist, gist)         
        
        for i in range(len(gist)):
            
            if (abs(gist[i] - grayHist[i]) <= gist[i]*0.07):
                resultGist += 1
            # какой критерий сравнение? 
            # необходима интерполяция гистограмм           
        
        print(resultGist, " " + name)  
        if(resultGist >= 90): 
            print(name)   
            return True
       
                             
        
    return False
                        
            
    
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
        #matches = bf.match(vector, des)
        #matches = sorted(matches, key = lambda x:x.distance)

        for m,n in matches:
            if m.distance < lowe_ratio*n.distance:
                goodDescript.append([m])
        #for i,m inна enumerate(matches):
            #if i < len(matches) - 1 and m.distance < lowe_ratio * matches[i+1].distance:
                #goodDescript.append([m])
        
        if(len(goodDescript) >= 65):
            print(datetime.now() - start_time)
            print(len(goodDescript))
            print(names[name_id])
            return True
        name_id += 1
    print(len(goodDescript))

    return False
        

# Функция основная обрабатывующая 
def searchLineObject(img, vectors, names, gistagram, gradients):
    global path, thresh 
    #cv.imshow("image", img)
    key = 0

    # Обработка изображения, изменение контрастности
    enhancer = ImageEnhance.Contrast(Image.fromarray(img, 'RGB'))
    enhancer_img = np.asarray(enhancer.enhance(1.5))
    #cv.imshow("img", enhancer_img)   
        
    imageGray = cv.cvtColor(enhancer_img, cv.COLOR_BGR2GRAY)  
    imgGray = imageGray.reshape(len(imageGray), len(imageGray[0]))
    imgGray = np.array([imgGray,imgGray,imgGray])
    imgGray = np.moveaxis(imgGray, 0, -1)
    
    #img_binary_1 = gs(np.asarray(imgGray), 2)
    gausimg = cv.GaussianBlur(imgGray, (5,5), 0)
    img_binary = cv.threshold(gausimg, thresh, 255, cv.THRESH_BINARY)[1]
            
    #cv.imshow("image", img_binary)

    #cv.imwrite(path+name, cv.threshold(img_binary, 165, 255, cv.THRESH_BINARY)[1])
    #cv.imwrite(path+name, img_binary)

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
        #output = img
    
        x,y,w,h = cv.boundingRect(c)
        #res_binary = img_binary[y:(y+h-30),x:(x+h-45)]
        res_img = img[y+25:(y+h-100),x+25:(x+h-120)]
        cv.imshow("result", res_img)
        res_grey_img = imgGray[y+25:(y+h-100),x+25:(x+h-120)]
        res_gaus_img = gausimg[y+25:(y+h-100),x+25:(x+h-120)]
        
        output = copy(img)
        output = cv.drawContours(output, c, -1, 255, 3)
        img_binary = cv.drawContours(img_binary, c, -1, 255, 3)
        cv.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow("cascade_image", output)
    #cv.imshow("res_gaus_img", res_gaus_img)
    (width, height) = res_img.shape[:2]
    cv.imshow("result_image", img_binary)
    #cv.imshow("result_image2", res_img)
    
       
    #if(width >= 200 and height >= 270):
    if(vector_comparison(vectors, res_gaus_img, names, 'ORB')):
        key = 0
        cv.destroyAllWindows()
        #cv.imshow("res_gaus_img", res_gaus_img)
        cv.imshow("result", res_img)
        cv.waitKey(0)  
        
        #if(detected_gistagram(res_img, gistagram, names)):
            #cv.destroyAllWindows()
            #cv.imshow("result", res_img)
            #cv.waitKey(0) 
            
        #if(detected_gradient(res_img, gradients, names)):
         #   cv.destroyAllWindows()
          #  cv.imshow("result", res_img)
           # cv.waitKey(0) 


    return key, res_img
    

#searchLineObject(cv.imread("Photo/236.jpg"), "0.jpg")
cv.waitKey()
cv.destroyAllWindows()

