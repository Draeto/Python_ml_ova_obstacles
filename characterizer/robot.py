from j2l.pyrobotx.robot import IRobot
from characterizer.interface import IPixelCharacterizer
from characterizer.pixel import GradPixelCharacterizer
from characterizer.image import ObstacleGradImageCharacterizer

class OvaLuminosityPixelCharacterizer(IPixelCharacterizer):
    def __init__(self, robot: IRobot):
        super().__init__()
        self.__robot = robot
    def characterize(self, x : int, y : int ) -> float:
        return self.__robot.getImagePixelLuminosity(x,y)

class OvaObstacleGradImageCharacterizer(ObstacleGradImageCharacterizer):
    def __init__(self, imageHeight:int, imageWidth:int, robot) -> None:
        super().__init__(
            imageHeight,
            imageWidth,
            GradPixelCharacterizer(OvaLuminosityPixelCharacterizer(robot))
        )