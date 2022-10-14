from copy import copy
from glob import glob
from posixpath import split
import threading
from time import sleep
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import json, codecs

path_to_vector = "OR_vector/"
path_to_gradient = "OR_gradient/"
path_to_gistagram = "OR_gistagram/"


def clearFilesForFolder(path):
    for file in glob(path):
        os.remove(file)
        

def calcGradients(img, format):
    global path_to_gradient
    
    I = cv.resize(img, dsize=[10,10])
    #gradient = np.gradient(copy(img))
    
    gradients = []
    list = []
    
    for i in range(10):
        for j in range(9):
            
            list.append(I[i][j])
            list.append(I[i][j+1])
            
            gradients.append([int(list[0][0]) - int(list[1][0]), int(list[0][1]) - int(list[1][1]), int(list[0][2]) - int(list[1][2])])
            #print(gradients[j])
            list.clear()
            
       
        
    with open(path_to_gradient +str(format[0]) + ".json", 'w', encoding='utf-8') as fw:
        json.dump(gradients, fw)
        

def calcGrayHist(I, format):
         # Рассчитать серую гистограмму
    global path_to_gistagram
    
    I = cv.resize(I, dsize=[20,20])
    
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    gistagram = grayHist
    
    with open(path_to_gistagram +str(format[0]) + ".json", 'w', encoding='utf-8') as fw:
        json.dump(gistagram.tolist(), fw)
        

def calcDescripters(image, format):
    global path_to_vector
    
    image = cv.resize(image, dsize=[265, 325])
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  
    image = cv.GaussianBlur(image, (5,5), 0)
    cv.imwrite("Photo/" + str(format[0] + ".jpg"), image)

    orb = cv.ORB_create()
    
    # Найти ключевые моменты
    kp = orb.detect(image, None)
    
    # вычисление дескрипторов
    kp, des = orb.compute(image, kp)
            
    with open(path_to_vector + str(format[0]) + ".json", 'w',  encoding='utf-8') as fw:
        json.dump(des.tolist(), fw)
        

def getDataForImage(img, filename):
    global path_to_vector, path_to_gradient, path_to_gistagram
    
    #gausimg = cv.GaussianBlur(img, (5,5), 0)
    image = img
    #image = cv.resize(img, dsize=[500,500])
    #image = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    #print(ntpath.dirname(img))
    
    #cv.imshow("image", image)
    
    #print(gradient.split('.'))
    # Инициация орб
    
    format = filename.split(".")
    
    threading.Thread(target= lambda: calcGrayHist(img, format), name='D').start()
    threading.Thread(target= lambda: calcGradients(img, format), name='F').start()
    threading.Thread(target= lambda:  calcDescripters(img, format), name='R').start()
    #calcGrayHist(img, format)
    #calcGradients(img, format)
    
    

"""    string = ""
    for i in des:
        string += str(i) + " "
    

    file = open(path_to_vector + str(format[0]) + ".txt", 'w')
    file.write(string)
    file.close()"""
    
    # нарисовать
    #img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    #cv.imshow("image", img2)
    #cv.waitKey(0)


#threading.Thread(target= lambda: clearFilesForFolder("OR_vector/*"), name='A').start()
#threading.Thread(target= lambda: clearFilesForFolder("OR_gradient/*"), name='B').start()
#threading.Thread(target= lambda: clearFilesForFolder("OR_gistagram/*"), name='C').start()
path = "image3/"
for filename in os.listdir(path):
    #img = cv.imread(os.path.join(path, filename))
    #img = cv.imdecode(np.fromfile(os.path.join(path, filename), dtype=np.uint8), cv.IMREAD_GRAYSCALE)      
    img = cv.imdecode(np.fromfile(os.path.join(path, filename), dtype=np.uint8), cv.IMREAD_COLOR)      
       
    if img is not None:
        getDataForImage(img, filename)


cv.destroyAllWindows()