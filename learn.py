import cv2 as cv
import numpy as np
import os
import json
import uuid
from PIL import Image


class Learning(object):
    
    __path_to_save = "Polus/data/" # базовый путь всех дескриптеров и других файлов
    __path_to_image_for_learn = "Polus/image/"
    _topic = "default/"
    descriptors = [] # список дескриптеров  
    names = [] # список имен существующих изображений
    unicode = [] # список юникодов дескриптеров   
    
    def descriptorLearning(self, pathForLearn: str = None, topic: str = None) -> list:
        
        """
        pathForLearn: откуда загружаются изображения
        topic: тема папки куда сохраняются файлы
        
        """        
        if(pathForLearn is not None):
            if(topic is not None):
                self._topic = topic
                return self.__loopForLearn(pathForLearn, topic)
            else:
                return self.__loopForLearn(pathForLearn, self._topic)
        else:
            if(topic is not None):
                self._topic = topic
                return self.__loopForLearn(self.__path_to_image_for_learn, topic) 
            else:
                return self.__loopForLearn(self.__path_to_image_for_learn, self._topic) 
            
    def __loopForLearn(self, pathForLearn: str, topic: str):
        
            pathTolearn = (self.__path_to_save + topic)
            pathToDes = (pathTolearn + "/" + "descript/")
            pathToName = (pathTolearn + "/" + "names/")
            pathToImage = (pathTolearn + "/" + "image/")       
            os.mkdir(pathTolearn)         
            os.mkdir(pathToDes)         
            os.mkdir(pathToName)         
            os.mkdir(pathToImage)         
            
            for filename in os.listdir(pathForLearn):
                img = cv.imdecode(np.fromfile(os.path.join(pathForLearn, filename), dtype=np.uint8), cv.IMREAD_COLOR)
                
                self.unicode.append(str(uuid.uuid4()).split("-")[-1])
                
                if img is not None:
                    self.descriptors.append(self.__learnDescripters(img, pathToDes, self.unicode[-1]))  
                    self.names.append(filename.split(".")[0])
                    
                    with open(pathToName + self.unicode[-1] + ".json", 'w',  encoding='utf-8') as fw:
                        json.dump(self.names[-1], fw)
                        
                    cv.imwrite(pathToImage + self.unicode[-1] + ".jpg", img)
                    
                    
            return self.descriptors, self.names
    
    def __learnDescripters(self, image: Image, path_to_vector: str, name: str) -> list:
        
        image = cv.resize(image, dsize=[265, 325])
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  
        image = cv.GaussianBlur(image, (5,5), 0)
        
        orb = cv.ORB_create()
        
        # Найти ключевые моменты
        kp = orb.detect(image, None)
        
        # вычисление дескрипторов
        kp, des = orb.compute(image, kp)
        
        with open(path_to_vector + name + ".json", 'w',  encoding='utf-8') as fw:
            json.dump(des.tolist(), fw)
        
        return des.tolist()