from copy import copy
from PIL import ImageEnhance
import numpy as np
import cv2 as cv
from PIL import Image

        
class Vision(object):
    
    __modeImageReturn = "OR" # выбор
    __lowe_ratio = 0.8 # параметр корректирующий разницу между дескрипторами, влияет на количетсво совпадающих точек. Чем меньше число тем меньше совпадений
    __thresh = 145 # граница отбора пикселей в бинарном формате, чем больше число тем больше будет отброшенно из кадра 
    __autoCal = False # Флаг для автоматического поиска дескриптеров после обработки изображений (НЕМНОГО ПОВЫШАЕТ ПРОИЗВОДИТЕЛЬНСТЬ ЕСЛИ TRUE) 
    descriptors = [] # список дескриптеров  
    names = [] # список имен существующих изображений
    unicode = [] # список юникодов дескриптеров
 
    
    def searchLineObject(self, img: Image,  autoComp: bool = None, mode: str = None, thresh: int = None) -> Image:
        """Управляющий метод для обработки изображений

        Args:
            img (Image): изображение загружаемое с OpenCv в формате numpy.array
            autoComp (bool, optional): Параметр отвечает за часть выделения объекта, чем он болье тем сильнее будет отсеена граница
            mode (string, optional): Данный параметр задаёт в каком формате вам вернуть изображение (1 канал или 3) defolt: OR (3 канала)
            thresh (int, optional): _description_. Defaults to None.

        Raises:
            RuntimeError: Error, variable passed incorrectly. You need to pass a Int variable!
            RuntimeError: Error, variable passed incorrectly. You need to pass a Boolean variable!
            RuntimeError: Error, check operation mode. It can only be OR or GR for a color or gray image, respectively
            RuntimeError: Error, check the input image
        Returns:
            Image: изображение в одном из двух вариантов (1 канал или 3 канала)
        """
        
        if(img is not None):
            if(thresh is not None):
                if(isinstance(thresh, int)):
                    self.__thresh = thresh
                else:
                    print("Error, variable passed incorrectly. You need to pass a Int variable!")
                    raise RuntimeError
            if(autoComp is not None):
                if(isinstance(autoComp, bool)):
                    self.__autoCal = autoComp
                else:
                    print("Error, variable passed incorrectly. You need to pass a Boolean variable!")
                    raise RuntimeError
                if (mode is None):
                    return self.__searchLineOject(img, self.__modeImageReturn)
                else:
                    if(mode == "OR" or "GR"):
                        return self.__searchLineOject(img, mode)
                    else:
                        print("Error, check operation mode. It can only be OR or GR for a color or gray image, respectively")
                        raise RuntimeError  
        else: 
            print("Error, check the input image.")
            raise RuntimeError                       
        # end def


    def __searchLineOject(self, img: Image, mode: str) -> Image:
            
            """
            Приватный метод который обрабатывает изображение для последующего анализа дескриптеров
            В этом методе повышается контрастность изобажения в 1,5 раза, проводится фильтрация гаусса
            после чего изображение бинаризируется и выделяется белая рамка в которой и будет объекс
            
            Метод имеет два режима работы gray и origin, каждый из которых возвращает изображения
            с определенным количеством каналов (1 или 3) 
            
            В автоматическом режими всегда возвращает серое изображение (1 канал) и сразу проводит расчёт
            дескриптеров и сравнивает их выводя сразу результат сравнения 
            
            Аргументы метода: 
            img - изображение в 3 канальном формате // image
            mode - режим возврата формата изображения (1 канал или 3)
            
            Возвращает:
            result_img - может быть как в серых оттенках так и классическое 3 канальное        
            """
            
            enhancer = ImageEnhance.Contrast(Image.fromarray(img, 'RGB'))
            enhancer_img = np.asarray(enhancer.enhance(1.5))
            
            imageGray = cv.cvtColor(enhancer_img, cv.COLOR_BGR2GRAY)  
            imgGray = imageGray.reshape(len(imageGray), len(imageGray[0]))
            imgGray = np.array([imgGray,imgGray,imgGray])
            imgGray = np.moveaxis(imgGray, 0, -1)
            
            gausimg = cv.GaussianBlur(imgGray, (5,5), 0)
            img_binary = cv.threshold(gausimg, self.__thresh, 255, cv.THRESH_BINARY)[1]
            
            lower = np.array([255, 255, 255], dtype="uint8")
            upper = np.array([255, 255, 255], dtype="uint8") 
                
            kernel = np.ones((15,15),np.uint8)
            img_morf = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel)
            mask = cv.inRange(img_morf, lower, upper)
            
            ret,thresh_img = cv.threshold(mask, self.__thresh, 255, 0)
            contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            if len(contours) != 0:    
                x,y,w,h = cv.boundingRect(max(contours, key = len))
                res_img = img[y+25:(y+h-110),x+25:(x+h-150)]
                res_grey_img = imgGray[y+25:(y+h-110),x+25:(x+h-150)] 
            else: 
                return None         

            if(self.__autoCal is True):
                return self.__desCom(res_grey_img, self.descriptors, self.names)        
            else:
                if(mode =="GR"): 
                    return res_grey_img
                else: return res_img            
            # end def   
            
    def __desCom(self, img: Image, datadescriptors: list, names: list) -> str:
            
            img = cv.resize(img, dsize=[265, 325])   # Подгоняем изображения под установленный размер         
            goodDescript = []
            name_id = 0
            finder = cv.ORB_create()
                
            kp = finder.detect(img,None)
            kp, des = finder.compute(img, kp)
            bf = cv.BFMatcher_create()
            
            for vector in datadescriptors:
                goodDescript.clear()
                vector = np.uint8(vector)
                matches = bf.knnMatch(des,vector, k=2)

                for m,n in matches:
                    if m.distance < self.__lowe_ratio*n.distance:
                        goodDescript.append([m])
                
                if(len(goodDescript) >= 35):
                    return names[name_id]
                name_id += 1
            return None
        # end def