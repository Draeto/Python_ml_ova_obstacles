from .interface import IPixelCharacterizer
from math import sqrt

class GradPixelCharacterizer(IPixelCharacterizer):
    def __init__(self, luminosityPixelCharacterizer: IPixelCharacterizer):
        super().__init__()
        self.__pixelCharacterizer = luminosityPixelCharacterizer

    def characterize(self, x :int, y : int) -> float:
        return sqrt(
            (self.__pixelCharacterizer.characterize(x + 1,y) - self.__pixelCharacterizer.characterize(x - 1, y)) **2
            +
            (self.__pixelCharacterizer.characterize(x, y + 1) - self.__pixelCharacterizer.characterize(x, y - 1)) **2
        )   