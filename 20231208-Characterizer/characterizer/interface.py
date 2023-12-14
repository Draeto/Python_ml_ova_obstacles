class IPixelCharacterizer:
    def characterize(self, x : int, y : int ) -> float:
       ...

class IImageCharacterizer :
    def characterize(self) -> list[float,float]:
       ...
