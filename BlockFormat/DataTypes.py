from enum import IntEnum
class DataType(IntEnum):
    NoData = 0
    EventData = 1
    NEURAL = 2
    MOTIONSENSOR = 3
    AUDIO = 4
    GPS = 7
    MULTIPLEMAGNETOMETER = 8
    ALTIMETER = 9