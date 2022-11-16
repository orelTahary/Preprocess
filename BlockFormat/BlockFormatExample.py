# This is a Python script demonstrating the extraction of data from block file format
import os
import HeaderConstants
import FileConstants
import numpy as np
import DataExtraction
import DataTypes
import MotionSensorConstants

# Set to your file path and file name
folder = '05.09.2022\\16'
file = 'NEUR0030.DF1'
fileName = os.path.join(folder, file)

# Set the following values from event file
# neural
numberOfAdcBits = 16
offset = 2 ** (numberOfAdcBits - 1)
voltageResolution = 1.95e-7
numberOfChannels = 64
neuralSamplingFrequency = 32000
# audio
numberOfAudioBits = 15
audioSamplingFrequency = 100000
isAudioSigned = True
# motion sensor
acclMax = 2*MotionSensorConstants.G  # m/s^2, maximum value of selected range
gyroMax = 250  # degrees per second,  maximum value of selected range
magMax = MotionSensorConstants.MAGNETOMETER_9250_RANGE  # Teslas,  maximum value of selected range

# read data from file as bytes
file = open(fileName, "rb")
byteData = file.read()
file.close()

# check blocks headers are present and report dropped blocks
blockStartIndices = []
droppedBlocks = []
byteData = np.frombuffer(byteData, 'uint8')
lastFilledBlock = DataExtraction.getLastBlockInFile(byteData)
for i in range(0, len(byteData), FileConstants.BLOCK_SIZE):
    thisBlockIdx = i/FileConstants.BLOCK_SIZE
    if thisBlockIdx > lastFilledBlock:
        break
    constId = byteData[i:i+len(HeaderConstants.HEX_CONST_ID)]
    if np.array_equal(constId, HeaderConstants.HEX_CONST_ID):
        blockStartIndices.append(i)
    elif np.array_equal(constId, FileConstants.EMPTY_BYTES_00) or np.array_equal(constId, FileConstants.EMPTY_BYTES_FF):
        droppedBlocks.append(thisBlockIdx)
    else:
        print("Warning: Unexpected values for header constant - file is corrupted")

for i in range(0, len(droppedBlocks)):
    print("Dropped block at block number {}".format(droppedBlocks[i]))

# extract and parse headers
# get timestamps
timestamps = []
for blockId in range(0, len(blockStartIndices)): #timestamps change in every header
    thisBlockStart = blockStartIndices[blockId]
    header = byteData[thisBlockStart:thisBlockStart + HeaderConstants.HEADER_TOTAL_BYTES]
    thisTimestamp = np.array(header[HeaderConstants.TIME_STAMP_POSITION:HeaderConstants.TIME_STAMP_POSITION +
                    HeaderConstants.TIME_STAMP_BYTES]).view(np.uint32)[0]
    timestamps.append(thisTimestamp)


# extract block structure
# structure of block is same in all blocks so using first header is sufficient
firstHeader = byteData[0:HeaderConstants.HEADER_TOTAL_BYTES]
partitionInfo = np.array(firstHeader[HeaderConstants.PARTITION_START_POSITION:HeaderConstants.PARTITION_START_POSITION + HeaderConstants.PARTITION_BYTES])
dataTypesPresent = []
dataStartIndices = []  # the byte within the block where this type of data starts
dataSegmentLengths = []
partitionIdx = 0
numBytes = 4  # stored as uint32
while (partitionIdx < len(partitionInfo)):
    dataTypesPresent.append(partitionInfo[partitionIdx:partitionIdx + numBytes].view(np.uint32)[0])
    partitionIdx += numBytes
    dataStartIndices.append(partitionInfo[partitionIdx:partitionIdx + numBytes].view(np.uint32)[0])
    partitionIdx += numBytes
    dataSegmentLengths.append(partitionInfo[partitionIdx:partitionIdx + numBytes].view(np.uint32)[0])
    partitionIdx += numBytes

# Extract neural data
IsNeuralPresent = DataTypes.DataType.NEURAL in dataTypesPresent  #check neural data is present
neuralData = []
if IsNeuralPresent:
    neuralBytes = DataExtraction.extractDataSegments(DataTypes.DataType.NEURAL, dataTypesPresent, blockStartIndices, byteData, dataStartIndices, dataSegmentLengths)
    neuralDataArray = DataExtraction.convertNeuralBytes(neuralBytes, voltageResolution, offset)

    for chanId in range(0, numberOfChannels):
        neuralData.append(neuralDataArray[chanId:len(neuralDataArray):numberOfChannels])

# Extract audio data
IsAudioPresent = DataTypes.DataType.AUDIO in dataTypesPresent
if IsAudioPresent:
    audioBytes = DataExtraction.extractDataSegments(DataTypes.DataType.AUDIO, dataTypesPresent,
                    blockStartIndices, byteData, dataStartIndices, dataSegmentLengths)
    audioData = DataExtraction.convertAudioData( isAudioSigned, audioBytes, numberOfAudioBits )


# Extract motion sensor data
IsMotionSensorPresent = DataTypes.DataType.MOTIONSENSOR in dataTypesPresent
if IsMotionSensorPresent:
    # get motion sensor bytes from file
    motionSensorBytes = DataExtraction.extractDataSegments(DataTypes.DataType.MOTIONSENSOR,
                       dataTypesPresent,blockStartIndices, byteData, dataStartIndices,
                       dataSegmentLengths)
    # convert to uint16
    motionSensorInts = np.array(motionSensorBytes).view(np.int16)
    # find block starts
    msStartBlockIndices = DataExtraction.extractMotionSensorBlockStarts(motionSensorInts,
                                           MotionSensorConstants.CONST_ID)
    # get motion sensor timestamp of each block
    motionSensorTimestamps = DataExtraction.extractMotionSensorTimestamps(motionSensorInts, msStartBlockIndices )
    # sort into accelerometer, gyro and magnetometer
    accelerometerInts, gyroscopeInts, magnetometerInts = DataExtraction.extractMotionSensor(
        motionSensorInts, msStartBlockIndices)

    # convert to physical units
    accelerometerData = DataExtraction.convertMotionSensorData(accelerometerInts,
                       MotionSensorConstants.ACCELEROMETER_NUMBER_OF_BITS, acclMax)
    gyroscopeData = DataExtraction.convertMotionSensorData(gyroscopeInts,
                       MotionSensorConstants.GYROSCOPE_NUMBER_OF_BITS, gyroMax)
    magnetometerData = DataExtraction.convertMotionSensorData(magnetometerInts,
                       MotionSensorConstants.MAGNETOMETER_NUMBER_OF_BITS, magMax)


    # sort data by axes
    xAccelerometer, yAccelerometer, zAccelerometer = DataExtraction.sortMotionSensorByAxes(
        accelerometerData, MotionSensorConstants.NUMBER_OF_AXES)
    xGyroscope, yGyroscope, zGyroscope = DataExtraction.sortMotionSensorByAxes(
        gyroscopeData, MotionSensorConstants.NUMBER_OF_AXES)
    xMagnetometer, yMagnetometer, zMagnetometer = DataExtraction.sortMotionSensorByAxes(
        magnetometerData, MotionSensorConstants.NUMBER_OF_AXES)

