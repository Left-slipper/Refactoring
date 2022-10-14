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

    # Инициация орб
    
    format = filename.split(".")

    threading.Thread(target= lambda:  calcDescripters(img, format), name='R').start()


path = "image3/"
for filename in os.listdir(path):   
    img = cv.imdecode(np.fromfile(os.path.join(path, filename), dtype=np.uint8), cv.IMREAD_COLOR)      
       
    if img is not None:
        getDataForImage(img, filename)


cv.destroyAllWindows()