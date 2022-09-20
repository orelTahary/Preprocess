import numpy as np
import FileConstants
import MotionSensorConstants
import DataTypes

def getLastBlockInFile(byteData): # this function checks the last block (the last file in a recording may not be filled until the end)
    expectedBlockStartIndices = []
    lastNonZero = np.max(np.nonzero(byteData)) #if empty bytes are set as 0
    lastNonFF = np.max(np.where(byteData == 255)) #if empty bytes are ff
    lastWrittenValue = np.max([lastNonFF, lastNonFF])
    for j in range(0, len(byteData), FileConstants.BLOCK_SIZE):
        thisBlockIdx = j / FileConstants.BLOCK_SIZE
        if lastWrittenValue > j: #if this block is non empty
            expectedBlockStartIndices.append(thisBlockIdx)

    lastFilledBlock = np.max(expectedBlockStartIndices)
    return lastFilledBlock

def extractDataSegments(dataType, dataTypesPresent, blockStartIndices, byteData, dataStartIndices, dataSegmentLengths):
    dataBytes = []
    segmentIdx = dataTypesPresent.index(dataType)  # check which data segment it is
    for blockId in range(0, len(blockStartIndices)):
        thisBlockStart = blockStartIndices[blockId]
        thisBlock = byteData[thisBlockStart:thisBlockStart + FileConstants.BLOCK_SIZE]
        dataBytes.extend(
            thisBlock[dataStartIndices[segmentIdx]:dataStartIndices[segmentIdx] + dataSegmentLengths[segmentIdx]])
    return dataBytes

def convertNeuralBytes(neuralBytes, voltageResolution, offset):
    neuralInts = np.array(neuralBytes).view(np.uint16)

    neuralData = voltageResolution*(neuralInts - float(offset))
    return neuralData

def convertAudioData( isAudioSigned, audioBytes, numberOfAudioBits ):
    if isAudioSigned:
        audioInts = np.array(audioBytes).view(np.int16)
        minValue = -2 ** (numberOfAudioBits - 1) + 1
        maxValue = 2 ** (numberOfAudioBits - 1)

    else:
        audioInts = np.array(audioBytes).view(np.uint16)
        minValue = 0
        maxValue = 2 ** numberOfAudioBits
    range = maxValue - minValue
    audioData = (audioInts - minValue) / float(range)
    return audioData

def extractMotionSensorByType(motionSensorInts, msStartBlockIndices, offsetPosition, segmentLength):
    offsets = motionSensorInts[msStartBlockIndices +
                                  offsetPosition]
    segmentSizes = motionSensorInts[msStartBlockIndices + segmentLength]
    thisTypeData = []
    for blockId in range(0, len(msStartBlockIndices)):
        startIdx = msStartBlockIndices[blockId] + offsets[blockId]
        thisAccelerometerData = motionSensorInts[startIdx:startIdx + segmentSizes[blockId]]
        thisTypeData.extend(thisAccelerometerData)
    return thisTypeData

def extractMotionSensorBlockStarts(motionSensorInts, constId):
    msStartBlockIndices = []
    for i in range(0, len(motionSensorInts)-1): # extract motion sensor block starts
        if (motionSensorInts[i] == constId[0]) & (motionSensorInts[i+1] == constId[1]):
            msStartBlockIndices.append(i)
    msStartBlockIndices = np.array(msStartBlockIndices)
    return msStartBlockIndices

def extractMotionSensor(motionSensorInts, msStartBlockIndices):

    accelerometerData = extractMotionSensorByType(motionSensorInts, msStartBlockIndices,
                      MotionSensorConstants.ACCELEROMETER_OFFSET_POSITION,
                      MotionSensorConstants.ACCELEROMETER_LENGTH_POSITION)
    gyroscopeData = extractMotionSensorByType(motionSensorInts, msStartBlockIndices,
                      MotionSensorConstants.GYROSCOPE_OFFSET_POSITION,
                      MotionSensorConstants.GYROSCOPE_LENGTH_POSITION)
    magnetometerData = extractMotionSensorByType(motionSensorInts, msStartBlockIndices,
                              MotionSensorConstants.MAGNETOMETER_OFFSET_POSITION,
                              MotionSensorConstants.MAGNETOMETER_LENGTH_POSITION)
    return accelerometerData, gyroscopeData, magnetometerData

def convertMotionSensorData(data, numberOfBits, maxValue):
    offset = 2 ** (numberOfBits - 1)
    scaledData = np.divide(data, float(offset)) * float(maxValue)
    return scaledData

def sortMotionSensorByAxes(data, numberOfAxes):
    xData = data[0::numberOfAxes]
    yData = data[1::numberOfAxes]
    zData = data[2::numberOfAxes]
    return xData, yData, zData

def extractMotionSensorTimestamps(motionSensorInts, msStartBlockIndices):
    timestamps = np.empty(np.size(msStartBlockIndices))
    timestampIndices = msStartBlockIndices + MotionSensorConstants.TIMESTAMP_INDEX[0]
    for timestampIdx in range(0, len(timestampIndices)):
        thisTimestamp = motionSensorInts[timestampIndices[timestampIdx]:timestampIndices[timestampIdx] + 2]
        thisTimestamp = np.array(thisTimestamp).view(np.uint32) * MotionSensorConstants.TIME_RESOLUTION
        timestamps[timestampIdx] = thisTimestamp
    return timestamps

