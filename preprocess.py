from typing import List

import numpy as np
import os
import scipy.signal as sig
from itertools import chain
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import logging
import sys
from datetime import datetime
import scipy.io as sio

__version__ = '0.0.5'


# def logging_test():
#     logging.debug('bli')
#     logging.warning('blibli')
# #
# Plot activity of electrodes in the wireless files

def plotWireless(fileName, plotLimit, channels, samplingRate=32000, nChannels=32, ):
    logging.info('started plotWireless function')
    if type(channels) is not list:
        channels = [channels]
        logging.debug('input was not in the format of a list, corrected')
    fig, axes = plt.subplots()
    plotRange = range(plotLimit[0] * samplingRate, plotLimit[1] * samplingRate)
    xRange: List[float] = [x / samplingRate for x in plotRange]
    with open(fileName, 'rb') as fid:
        allChannels = np.fromfile(fid, dtype=np.uint16)
    for channelNum in channels:
        channel = (allChannels[channelNum - 1::nChannels] - 32768).astype(np.int16)
        axes.plot(xRange, channel[plotRange], label=f'Elec {channelNum}')
    axes.set_xlabel('Time (s)')
    axes.set_title(f'File {fileName} Electrode {channelNum}')
    axes.legend(loc='upper left')
    return fig, axes


def plotAO(fileDir, filePrefix, fileList, plotLimit, channels, samplingRate=44000, nChannels=32, ):
    logging.info('started plotAO function')
    if type(channels) is not list:
        channels = [channels]
        logging.debug('input was not in the format of a list, corrected')
    fig, axes = plt.subplots()
    plotRange = range(plotLimit[0] * samplingRate, plotLimit[1] * samplingRate)
    xRange: List[float] = [x / samplingRate for x in plotRange]
    for elecNum in channels:
        logging.info(f'Processing electrode {elecNum}')
        elecName = f'CRAW_{elecNum:03d}'
        elecData = [None] * len(fileList)
        for i, fileNum in enumerate(fileList):
            fileName = f'{fileDir}{filePrefix}{fileNum:04d}.mat'
            matList = sio.loadmat(fileName, variable_names=elecName)
            elecData[i] = matList[elecName][0, :]
        allData = np.concatenate(elecData)
        axes.plot(xRange, allData[plotRange], label=f'Elec {elecNum}')
    axes.set_xlabel(f'Time (s)')
    axes.set_title(f'AO files in {fileDir}, channels: {channels}')
    axes.legend(loc='upper left')
    return fig, axes


#
# Transform wireless files to concatenated binary, single electrode, files
#
def wirelessToBin(inDir, outDir, files, elecList, nChannels=32, verbose=False):
    logging.info('started wirelessToBin function')
    nSamples = 0
    safeOutputDir(outDir)
    # Open all the output files
    rangeStr = "-F{0}T{1}".format(files[0], files[-1])
    ofids = openFids(outDir, elecList, "Elec{0}" + rangeStr + ".bin", "wb")
    # Read each wireless file and separate to electrodes
    for file in files:
        fileName = "{0}NEUR{1}{2}.DT2".format(inDir, '0' * (4 - len(str(file))), file)
        if not os.path.isfile(fileName):
            fileName = "{0}BACK{1}{2}.DT2".format(inDir, '0' * (4 - len(str(file))), file)
        if verbose:
            print(f'Transforming {fileName} to binary')
        fid = open(fileName, 'rb')
        fileData = np.fromfile(fid, dtype=np.uint16)
        nSamples += fileData.shape[0] // nChannels
        for elec in elecList:
            try:
                channelData = (fileData[elec - 1::nChannels] - 32768).astype(np.int16)
            except ValueError as err:
                logging.error('channel or file defined as input does not exist')
            channelData.tofile(ofids[elec])
        fid.close()
    # Close all the output files
    closeFids(ofids, elecList)
    return nSamples


def AOtoBin(fileDir, filePrefix, fileList, elecList, saveLfp=True, saveRaw=True, saveFilter=True, sampRate=44000,
            lfpBand=[2, 300], filterBand=[300, 6000]):
    bPass, aPass = sig.butter(4, [300 / (sampRate / 2), 6000 / (sampRate / 2)], btype='bandpass')
    # [bf, af] = sig.butter(4, [f/(1000/2) for f in freq], btype='band')
    bNotch, aNotch = sig.iirnotch(50 / (1000 / 2), 30)
    for elecNum in elecList:
        elecName = f'CRAW_{elecNum:03d}'
        logging.info(f'Processing electrode: {elecName}')
        if saveRaw:
            if not os.path.exists(os.path.join(fileDir, 'Raw')):
                os.mkdir(os.path.join(fileDir, 'Raw'))
            rawDir = os.path.join(fileDir, 'Raw', "")
            outRawFileName = f'{rawDir}{filePrefix}Raw{elecNum:03d}-{fileList[0]}-{fileList[-1]}.bin'
        if saveFilter:
            if not os.path.exists(os.path.join(fileDir, 'Filter')):
                os.mkdir(os.path.join(fileDir, 'Filter'))
            fiterDir = os.path.join(fileDir, 'Filter', '')
            outFilterFileName = f'{fiterDir}{filePrefix}Filter{elecNum:03d}-{fileList[0]}-{fileList[-1]}.bin'
        if saveLfp:
            if not os.path.exists(os.path.join(fileDir, 'Lfp')):
                os.mkdir(os.path.join(fileDir, 'Lfp'))
            lfpDir = os.path.join(fileDir, 'Lfp', '')
            outLfpFileName = f'{lfpDir}{filePrefix}Lfp{elecNum:03d}-{fileList[0]}-{fileList[-1]}.bin'
        elecData = [None] * len(fileList)
        for i, fileNum in enumerate(fileList):
            fileName = f'{fileDir}{filePrefix}{fileNum:04d}.mat'
            logging.info(f'Processing electrode: {fileName}')
            matList = sio.loadmat(fileName, variable_names=elecName)
            elecData[i] = matList[elecName][0, :]
        allData = np.concatenate(elecData)
        allData = allData.astype(np.int16)
        if saveRaw:
            allData.tofile(outRawFileName)
        if saveFilter:
            filtData = sig.filtfilt(bPass, aPass, allData)
            filtData.astype(np.int16).tofile(outFilterFileName)
        if saveLfp:
            lfpData = sig.decimate(allData, int(sampRate / 1000), ftype='fir')
            lfpData = sig.filtfilt(bNotch, aNotch, lfpData)
            lfpData.astype(np.int16).tofile(outLfpFileName)


#
# Plot activity of electrode in a bin file
#
def plotBin(fileName, plotLimit, samplingRate=32000, axes=None):
    logging.info("started plotBin function")
    if not axes:
        fig, axes = plt.subplots()
    plotRange = range(int(plotLimit[0] * samplingRate), (int(plotLimit[1] * samplingRate) - 1))
    xRange = [x / samplingRate for x in plotRange]
    try:
        with open(fileName, 'rb') as fid:
            channel = np.fromfile(fid, dtype=np.int16)
            axes.plot(xRange, channel[plotRange])
            axes.set_xlabel('Time (s)')
            axes.set_title(f'File {fileName}')
    except IOError:
        logging.warning(f'Unable to open file: {fileName}')
    if 'fig' in locals():
        return fig, axes
    return axes


#
# Iterate over binary files and remove the median
#
def remMedian(inDir, outDir, elecList, rangeStr, batchSize=100000, verbose=False):
    logging.info("started remMedian function")
    nElecs = len(elecList)
    safeOutputDir(outDir)
    # Open all the input and output files
    fileName = "{0}Elec{1}-F0T{2}.bin"
    ifids = openFids(inDir, elecList, 'Elec{0}' + rangeStr + '.bin', "rb")
    ofids = openFids(outDir, elecList, 'Elec{0}' + rangeStr + '.bin', "wb")

    if verbose:
        logging.info(f'File size {int(os.fstat(ifids[elecList[0]].fileno()).st_size / np.int16().itemsize)} samples')

    # Remove median from each channel
    location, readMore = 0, True
    inBuffer = np.zeros((nElecs, batchSize), dtype=np.int16)
    while readMore:
        if verbose:
            print(f'Location {location}')
        for i, elec in enumerate(elecList):
            data = np.fromfile(ifids[elec], count=batchSize, dtype=np.int16)
            if i == 0 and data.shape[0] != batchSize:
                inBuffer = np.zeros((nElecs, data.shape[0]), dtype=np.int16)
                readMore = False
            inBuffer[i, :] = data
        for i, elec in enumerate(elecList):
            notElec = list(range(0, i)) + list(range(i + 1, nElecs))
            outBuffer = inBuffer[i, :] - np.median(inBuffer[notElec, :], axis=0)
            outBuffer.astype(np.int16).tofile(ofids[elec])
        location += data.shape[0]

    # Close all the input and output files
    closeFids(ifids, elecList)
    closeFids(ofids, elecList)


#
# Transform wireless data to motion data
#
def wirelessToMotion(inDir, files, outDir=None, verbose=False, samplingRate=32000):
    logging.info("started wirelessToMotion function")
    motionData = np.empty(shape=0)
    # Read each wireless file
    for file in files:
        fileName = '{0}NEUR{1}{2}.DT2'.format(inDir, '0' * (4 - len(str(file))), file)
        if verbose:
            print(f'Read raw file {fileName}')
        fid = open(fileName, 'rb')
        fileData = np.fromfile(fid, dtype=np.int16)
        motionData = np.concatenate((motionData, fileData[0::32]))
        fid.close()

    bBlock = range(0, motionData.shape[0], 1024)
    nBlock = len(bBlock)
    print(f'Total data length: {motionData.shape[0]} Number of blocks {nBlock}')
    blockIndex = {"acc": (2, 6, np.zeros((nBlock, 2), dtype=np.int32)),
                  "gyr": (3, 7, np.zeros((nBlock, 2), dtype=np.int32)),
                  "mag": (4, 8, np.zeros((nBlock, 2), dtype=np.int32))}
    for i, block in enumerate(bBlock):
        if motionData[block] != 13579 or motionData[block + 1] != 24680:
            print('Error in block {0} (#1 {1} #2 {2})'
                  .format(i, motionData[block], motionData[block + 1]))
        else:
            for index in blockIndex:
                # print(motionData[block+blockIndex[index][0]])
                blockIndex[index][2][i, 0] = motionData[block + blockIndex[index][0]]
                blockIndex[index][2][i, 1] = motionData[block + blockIndex[index][1]]
    totalLen = {index: int(np.sum(blockIndex[index][2][:, 1]) / 3) for index in blockIndex}

    sensors = ['acc', 'gyr', 'mag']
    axes = ['x', 'y', 'z']
    kinematics, bk = {}, {}
    for sensor in sensors:
        kinematics[sensor] = {}
        bk[sensor] = 0
        for axis in axes:
            kinematics[sensor][axis] = np.zeros(totalLen[sensor])

    for i, b in enumerate(bBlock):
        for sensor in sensors:
            data = motionData[b + blockIndex[sensor][2][i, 0]:
                              b + blockIndex[sensor][2][i, 0] + blockIndex[sensor][2][i, 1]]
            ldata = int(data.shape[0] / 3)
            for j, axis in enumerate(axes):
                kinematics[sensor][axis][bk[sensor]:bk[sensor] + ldata] = data[j:data.shape[0]:3]
            bk[sensor] += ldata

    df = pd.DataFrame(columns=pd.MultiIndex.from_product([sensors, axes],
                                                         names=['sensorName', 'sensorNum']))
    msElecData = motionData.shape[0] * 1000 // samplingRate
    for sensor in sensors:
        for axis in axes:
            msMotionData = kinematics[sensor][axis].shape[0]
            if msMotionData != msElecData:
                print(f'Adjust ASYNC {sensor} {axis} acquisition data({msElecData}ms) != motion({msMotionData}ms)')
                kinematics[sensor][axis] = sig.resample(kinematics[sensor][axis], msElecData).astype(np.int16)
            df[sensor, axis] = kinematics[sensor][axis]

    if outDir is not None:
        safeOutputDir(outDir)
        fileName = "{0}Motion-F{1}T{2}.pkl".format(outDir, files[0], files[-1])
        print(fileName)
        with open(fileName, 'wb') as file:
            pickle.dump(df, file)

    return df


#
# Transform wireless data to motion data
#
def binToLFP(inDir, outDir, filePattern, elecList, freq=[2, 300], notch=False, verbose=False):
    logging.info("started binToLFP function")
    safeOutputDir(outDir)
    [bf, af] = sig.butter(4, [f / (1000 / 2) for f in freq], btype='band')
    for elec in elecList:
        inFileName = filePattern.format(inDir, str(elec))
        ifid = open(inFileName, 'rb')
        data = np.fromfile(ifid, dtype=np.int16)
        ifid.close()
        sdata = sig.resample(data, num=data.shape[0] // 32)
        sdata = sig.filtfilt(bf, af, sdata).astype(np.int16)
        if notch:
            F0, Q, Fs = 50, 35, 1000
            [bcomb, acomb] = sig.iirnotch(F0, Q, Fs)
            sdata = sig.filtfilt(bcomb, acomb, sdata).astype(np.int16)
        outFileName = filePattern.format(outDir, str(elec))
        ofid = open(outFileName, 'wb')
        sdata.tofile(ofid)
        if verbose:
            print(f'Tranform binary file {inFileName} to LFP file {outFileName}')


def openFids(dirName, elecList, filePattern, mode):
    fids = {}
    for elec in elecList:
        fileName = os.path.join(dirName, filePattern.format(str(elec)))
        try:
            fids[elec] = open(fileName, mode)
        except OSError:
            print("Cannot open file {0}".format(fileName))
            return {}
    return fids


def closeFids(fids, elecList):
    for elec in elecList:
        fids[elec].close()


def safeOutputDir(outDir):
    if outDir is not None and not os.path.isdir(outDir):
        try:
            os.mkdir(outDir)
        except OSError:
            print("Creation of the output directory {0} failed".format(outDir))


def bandpass_filter(inDir, outDir, filePattern, elecList, freq=[300, 6000], notch=False, verbose=False, Fs=32000):
    logging.info("started bandpass filter function")
    safeOutputDir(outDir)
    [bf, af] = sig.butter(4, [f / (Fs / 2) for f in freq], btype='band')
    for elec in elecList:
        inFileName = os.path.join(inDir, filePattern.format(str(elec)))
        ifid = open(inFileName, 'rb')
        data = np.fromfile(ifid, dtype=np.int16)
        ifid.close()
        sdata = sig.filtfilt(bf, af, data).astype(np.int16)
        if notch:
            F0, Q = 50, 35
            [bcomb, acomb] = sig.iirnotch(F0, Q, Fs)
            sdata = sig.filtfilt(bcomb, acomb, sdata).astype(np.int16)
        outFileName = os.path.join(outDir, filePattern.format(str(elec)))
        ofid = open(outFileName, 'wb')
        sdata.tofile(ofid)
        if verbose:
            print(f'Tranform binary file {inFileName} to LFP file {outFileName}')


def plot_corr_mat(dataDir, rangeStr, file_list, raw_fold='binNew', filt_fold='binBand', draw_lfp=False):
    rawRange = [os.path.join(dataDir, raw_fold, f"Elec{i}{rangeStr}.bin") for i in file_list]
    offset = 3200 * 90
    count = 3200 * 240
    samplingRate = 3200

    elec_array = np.zeros((len(file_list), count))
    for i, f in enumerate(rawRange):
        elec_array[i, :] = np.fromfile(f, dtype=np.int16, count=count)

    # print(os.path.isdir(os.path.join(dataDir, filt_fold)))
    if not os.path.isdir(os.path.join(dataDir, filt_fold)):
        bb, ab = sig.butter(4, [300 / (samplingRate / 2), 6000 / (samplingRate / 2)], btype='bandpass')
        filt_array = sig.filtfilt(bb, ab, elec_array)
    else:
        filt_array = np.zeros((len(file_list), count))
        filtRange = [os.path.join(dataDir, filt_fold, f"Elec{i}{rangeStr}.bin") for i in file_list]
        for i, f in enumerate(filtRange):
            elec_array[i, :] = np.fromfile(f, dtype=np.int16, count=count)

    if draw_lfp:
        bl, al = sig.butter(4, 50 / (samplingRate / 2), btype='lowpass')
        lfp_array = sig.filtfilt(bl, al, elec_array)
        lfp_array = lfp_array[:, 1000:]
        ccl = np.corrcoef(lfp_array)

    elec_array = elec_array[:, 1000:]
    filt_array = filt_array[:, 1000:]

    ccr = np.corrcoef(elec_array)
    ccf = np.corrcoef(filt_array)
    if draw_lfp:
        fig, ax = plt.subplots(figsize=(30, 10), nrows=1, ncols=3, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(30, 10), nrows=1, ncols=2, sharex=True)

    cax = ax[0]
    csr = cax.imshow(ccr)
    cax.set_title(f'Raw correlation matrix')
    cax.set_xticks(range(0, len(file_list)))
    cax.set_xticklabels(file_list)
    cax.set_yticks(range(0, len(file_list)))
    cax.set_yticklabels(file_list)
    fig.colorbar(csr, ax=ax[0])

    cax = ax[1]
    csf = cax.imshow(ccf)
    cax.set_title(f'Spike correlation matrix')
    cax.set_xticks(range(0, len(file_list)))
    cax.set_xticklabels(file_list)
    cax.set_yticks(range(0, len(file_list)))
    cax.set_yticklabels(file_list)
    fig.colorbar(csf, ax=ax[1])
    if draw_lfp:
        cax = ax[2]
        csl = cax.imshow(ccl)
        cax.set_title(f'LFP correlation matrix')
        cax.set_xticks(range(0, len(file_list)))
        cax.set_xticklabels(file_list)
        cax.set_yticks(range(0, len(file_list)))
        cax.set_yticklabels(file_list)
        fig.colorbar(csl, ax=ax[2])

    return ccr, ccf


def plot_channels(dataDir, fileList, elecList, num_seconds_to_plot=5, samplingRate=32000, bs=900000, ylim=[-700, 700],
                  raw_fold='binNew'):
    fig, ax = plt.subplots(figsize=(20, len(elecList) * 4), nrows=len(elecList) * 2, ncols=1, sharex=True, sharey=True)
    try:
        plt.ylim(ylim)
    except:
        print(f'y axis not limited, possibly wrong input')
    num_seconds_to_plot = 5
    samplingRate = 32000
    bs = 900000
    be = int(bs + samplingRate * num_seconds_to_plot)
    t = np.arange(bs, be, 1) / samplingRate
    bb, ab = sig.butter(4, [300 / (samplingRate / 2), 6000 / (samplingRate / 2)], btype='bandpass')
    for i, elc in enumerate(elecList):
        rangeStr = "-F{0}T{1}".format(fileList[0], fileList[-1])
        fpath = os.path.join(dataDir, raw_fold, 'Elec' + str(elc) + rangeStr + '.bin')
        elec_data = np.fromfile(fpath, dtype=np.int16)
        spk_data = sig.filtfilt(bb, ab, elec_data)
        ax[i * 2].plot(t, elec_data[bs:be])
        ax[i * 2].set_title(f'Raw data channel {elc}')
        ax[i * 2 + 1].plot(t, spk_data[bs:be])
        ax[i * 2 + 1].set_title(f'Spiking data channel {elc}')

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', filename='preprocess log', level=logging.DEBUG)
    logging.debug('using main of preprocessing')
    remMedian('/mnt/hgfs/vmshared/WLnew/', '/mnt/hgfs/vmshared/WLnew/out/',
              list(chain(range(2, 15), range(17, 19), range(20, 32))), batchSize=100000, verbose=True)

# "{0}Elec{1}Motion.bin"
# wirelessToMotion('/mnt/hgfs/vmshared/WLnew/','/mnt/hgfs/vmshared/WLnew/bin/',list(range(99,150)))

# wirelessToBin('/mnt/hgfs/vmshared/WLnew/','/mnt/hgfs/vmshared/WLnew/binNew/',list(range(99,102)),[3,9,31])
