from .interface import IImageCharacterizer
from .pixel import IPixelCharacterizer 
from math import sqrt

class ObstacleGradImageCharacterizer(IImageCharacterizer):
    def __init__(self, imageHeight:int, imageWidth:int, gradCharacterizer: IPixelCharacterizer) -> None:
        super().__init__()
        self.__imageHeight = imageHeight
        self.__imageWidth = imageWidth
        self.__pixelCharacterizer = gradCharacterizer

    def characterize(self) -> list[float, float] :
        # Calcul de la moyenne des gradients de l'image 
        moyenne = 0
        for x in range (self.__imageWidth):
            for y in range (self.__imageHeight):
                moyenne += self.__pixelCharacterizer.characterize(x, y)
        moyenne /= self.__imageWidth*self.__imageHeight

        # Calcul de l'ecart-type des gradients de tous les points de l'image
        std=0
        for x in range (self.__imageWidth):
            for y in range (self.__imageHeight):
                std+=(moyenne - self.__pixelCharacterizer.characterize(x, y)) **2
        std /= self.__imageWidth*self.__imageHeight
        std = sqrt(std)

        # Renvoie de la liste des variables explicatives
        return[moyenne, std]
    

class ObstacleGradHalfImageCharacterizer(IImageCharacterizer):
    def __init__(self, imageHeight:int, imageWidth:int, gradCharacterizer: IPixelCharacterizer) -> None:
        super().__init__()
        self.__imageHeight = imageHeight
        self.__imageWidth = imageWidth
        self.__pixelCharacterizer = gradCharacterizer

    def characterize(self) -> list[float, float] :
        # Calcul de la moyenne des gradients de l'image 
        moyenne, n = 0 , 0
        for x in range (self.__imageWidth):
            for y in range (self.__imageHeight // 2, self.__imageHeight):
                moyenne += self.__pixelCharacterizer.characterize(x, y)
                n += 1
        moyenne /= n

        # Calcul de l'ecart-type des gradients de tous les points de l'image
        std=0
        for x in range (self.__imageWidth):
            for y in range (self.__imageHeight // 2, self.__imageHeight):
                std+=(moyenne - self.__pixelCharacterizer.characterize(x, y)) **2
        std /= n
        std = sqrt(std)

        # Renvoie de la liste des variables explicatives
        return[moyenne, std]