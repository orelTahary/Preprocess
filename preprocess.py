from typing import List

import numpy as np
import numbers
import os
import scipy.signal as sig
from scipy.stats import pearsonr
from scipy.optimize import minimize
from itertools import chain
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import logging
import re
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
    if not (outDir is None):
        safeOutputDir(outDir)
    # Open all the output files
    if not (outDir is None):
        rangeStr = "-F{0}T{1}".format(files[0], files[-1])
        ofids = openFids(outDir, elecList, "Elec{0}" + rangeStr + ".bin", "wb")
    # Read each wireless file and separate to electrodes
    for file in files:
        fileName = file
        if type (fileName) != str:
            fileName0 = f'NEUR{file:04d}.DT2'
            fileName1 = f'BACK{file:04d}.DT2'
        else:
            fileName0 = fileName
        if not (inDir is None):
            fileName0 = os.path.join (inDir, fileName0)
            fileName1 = os.path.join (inDir, fileName1)
        # fileName = "{0}NEUR{1}{2}.DT2".format(inDir, '0' * (4 - len(str(file))), file)
        if os.path.isfile(fileName0):
            fileName = fileName0
        else:
            fileName = fileName1
            # fileName = "{0}BACK{1}{2}.DT2".format(inDir, '0' * (4 - len(str(file))), file)
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
            if not (outDir is None):
                channelData.tofile(ofids[elec])
        fid.close()
    # Close all the output files
    if not (outDir is None):
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


# The function plots a whole electrode Data set
def plotAllBin(fileName, elecNumber, samplingRate=32000, axes=None):
    logging.info("started plotBin function")
    if not axes:
        fig, axes = plt.subplots()
    try:
        with open(fileName, 'rb') as fid:
            channel = np.fromfile(fid, dtype=np.int16)
            axes.plot(channel)
            axes.set_xlabel('Time (s)')
            axes.set_title("All electrode " + elecNumber, fontsize=20)
            fig.set_size_inches((30, 5))
    except IOError:
        logging.warning(f'Unable to open file: {fileName}')
    if 'fig' in locals():
        return fig, axes
    return axes


# The function shows a single electrode data by path
def showElectrode(path, elecNumber, plotLimit, title):
    fig, axes = plotBin(path, plotLimit)
    axes.set_title('Electrode ' + str(elecNumber) + title, fontsize=20)
    fig.set_size_inches((30, 5))
    plt.show()


# The function plot 5 seconds from the start, middle and ending of the data set
def plot5Sec(filePath, axes, elec, samplingRate=32000):
    # finds max X value of the data set
    maxX = axes[1].dataLim.intervalx[1]
    # divide by sampling rate to get the time
    time = maxX / samplingRate
    # show start middle and end
    start = [0, 5]
    end = [time - 5, time]
    middle = [(time / 2) - 2.5, (time / 2) + 2.5]
    showElectrode(filePath, elec, start, " Start")
    showElectrode(filePath, elec, middle, " Middle")
    showElectrode(filePath, elec, end, " End")


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


def raw_to_noise_correlation(k, signal, sigMedian):
    return np.sum((signal - k * sigMedian) ** 2)


def find_k(signal, sigMedian):
    best_k = minimize(raw_to_noise_correlation, 0, args=(signal, sigMedian))
    return best_k


#
# Iterate over binary files and remove the median multiplied by a scalar
#
def remScaledMedian(inDir, outDir, elecList, rangeStr, batchSize=100000, verbose=False):
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
    first = True
    scalars = []
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
            if first:
                scalars.append(find_k(inBuffer[i, :], np.mean(inBuffer[notElec, :], axis=0)).x)
            outBuffer = inBuffer[i, :] - scalars[i] * np.mean(inBuffer[notElec, :], axis=0)
            outBuffer.astype(np.int16).tofile(ofids[elec])
        if first:
            first = False
            print(scalars)
        location += data.shape[0]

    # Close all the input and output files
    closeFids(ifids, elecList)
    closeFids(ofids, elecList)


#
# Transform wireless data to motion data
#
DATAFILELEN = 16*1024*1024
DATANCHANNELS=32
DATAFREQ=32000
DMCHANNEL=0
DMSIGNATURE = np.array ([13579, 24680])
DMTSFACTOR = 16
DMSAMPLERATE= 1000 # Hz
DATAMAGNETSAMPLERATE= 111 # Hz
DMHEADERSIZE = 12
BLOCKLEN = 64*1024
BLOCKSIGNATURE = [
    0x90ef, 0x5678, 0xabcd, 0x1234, # Signature
    0x1, 0x0,                       # Format version.
    0x0, 0x1                        # Size of block (64KB)
]
NEURONTYPE = 2
MOTIONTYPE = 3

# Find runs of ones. See
# https://newbedev.com/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encoding
#
# Parameters:
# bits - An arrray of 0-s and 1-s.
#
# Returns:
# start, finish, length - Start, finish and length of runs of 1-s in bits array.
#
def runs_of_ones_array (bits):
    """
Find runs of ones.

Parameters:
bits - An arrray of 0-s and 1-s.

Returns:
start, finish, length - Start, finish and length of runs of 1-s in bits array.
    """
    # Make sure array is bound by 0-s.
    bounded = np.concatenate (([0], bits, [0]))
    # Run start is +1, run end is -1.
    diffs = np.diff (bounded)
    # Compute lengths.
    begin, = np.where (diffs > 0)
    finish, = np.where (diffs < 0)
    return begin, finish, finish-begin

#
# Transform wireless data to neural data
#
def wirelessToChannels (base, files, prefix='NEUR',
                        verbose=False, nchannels=DATANCHANNELS, 
                        freq=DATAFREQ, savetype=np.int16):
    logging.info("started wirelessToMotion function")
#    sensors = ['acc', 'gyr', 'mag']
#    axes = ['x', 'y', 'z']
#    files = (f if type (f) is str else f'{prefix}{f:04d}.DF1'
#             for f in files)
#    if not base is None:
#        files = (os.path.join (base, datafile) for datafile in files)

    channeldata = []
    btimestamps = []
    for datafile in files:
        if verbose:
            print(f'Read raw file {datafile}')

        fd = open(datafile, 'rb')
        data = np.fromfile(fd, dtype=np.uint16)
        fd.close()

        #
        # Generic Stage
        #
        # Data files should be fixed length.
        if len (data)*2 != DATAFILELEN:
            raise Exception (f'File {datafile} size is {len (data)*2}!')

        # File is composed of 64KB blocks.
        blocks = data.reshape (-1, BLOCKLEN // 2)
        # Test blocks signatures.
        goodblocks = np.all (blocks [:, :len(BLOCKSIGNATURE)] == BLOCKSIGNATURE,
                             axis=1)
        # Extract timestamps.
        timestamps = np.dot (blocks [:, 8:10], [[1], [2**16]])
        timestamps = timestamps.astype (np.uint32).reshape (-1)
        # Extract data partitions.
        # Partition info are 3 uint32: Type, Start, Length.
        nblocks = len (blocks)
        compmat = (np.diag (np.ones (7*6)) [::2] +
                   np.diag (np.ones (7*6)) [1::2] * 2**16).T
        partinfo = np.dot (blocks [:, 12:54], compmat).reshape (nblocks, -1, 3)
        # Get reference to neuronal data.
        ind0, ind1 = np.where (partinfo [goodblocks, :, 0] == 2)
        if np.any (np.unique (ind0, return_counts=True) [1] > 1):
            raise f'Non unique neuronal data partition in {datafile}'

        # Extract the data.
        ilast, plast = ind0 [-1], ind1 [-1]
        t0, tn = timestamps [ind0 [0]], timestamps [ilast]
        if len (btimestamps) > 0 and btimestamps [-1] [1] != t0:
            if verbose:
                logging.warning (f'Timestamp discontinuity between files '+
                                 f'{btimestamps [-1] [1]}..{t0})')
                logging.warning ('Adjusting blocks.')
            diff = t0 - btimestamps [-1] [1]
            if diff > 0:
                # Pad with zeros
                channeldata.append (np.zeros (
                    (int (diff * freq // 1000), nchannels)
                ))
            else:
                # Truncate previous data.
                index = len (channeldata)-1
                while diff > 0 and index >= 0:
                    trunc = int (diff * freq // 1000)
                    if trunc > len (channeldata [index]):
                        trunc = len (channeldata [index])
                    channeldata [index] = channeldata [index] [:trunc]
                    index -= 1
        #
        # Neuronal Data Handling
        #
        channels = np.zeros ((int ((tn - t0) * freq // 1000 +
                                   partinfo [ilast, plast, 2] // (2*nchannels)),
                              nchannels), dtype=savetype)
        prevend = -1
        for i0, i1, ti in zip (ind0, ind1, timestamps [ind0]):
            _, start, length = partinfo [i0, i1]
            # Timestamp is too coarse, but we assume it's precise. It may cause
            # a glitch once in a while, but it seems the data is synchronised to
            # the timestamp.
            cs = (ti-t0) * int(freq // 1000)
            ce = cs + int (length // (2*nchannels))
#            print (cs, ce, ce - cs, length // (2*nchannels))
            # print (channels [cs:ce, :].shape)
            # print (blocks [i0, int (start // 2):int ((start + length) // 2)]
            #     .reshape (-1, nchannels).shape)
            channels [cs:ce, :] = (
                blocks [i0, int (start // 2):int ((start + length) // 2)]
                .reshape (-1, nchannels) - 2**15
                ).astype (savetype)
            prevend = ti + (length // (2*nchannels*(freq // 1000)))
        channeldata.append (channels)
        btimestamps.append ([t0, prevend])
    for i in range(len(channeldata)):
        channeldata[i] = channeldata[i].astype(savetype)
    return np.concatenate (channeldata).astype(savetype), btimestamps

def getDataFiles (inDir, files, prefix=['NEUR'], suffix='DF1', verbose=False):
    """
Get the data files from the file list.

Parameters:
inDir - Directroy where the files reside (None only uses file path in files).
files - A list of file names or file numbers.
prefix - List of filename prefix. When files are provided as numbers, the prefix
         list is used to search for the file in inDir. First matched file is
         used as the file.
suffix - Data files suffix. Used to construct filename when files are provided
         as numbers.
verbose - Whether to use verbose logging. At this time ignored.

Returns:
A list of valid data file paths.

Note:
If a file is not found, a warning is logged, but processing continues.
    """
    filepaths = []
    for f in files:
        if type (f) is str:
            pass
        elif isinstance (f, numbers.Real):
            alternatives = (os.path.join (inDir, f'{pfx}{f:04d}.{suffix}')
                            for pfx in prefix)
            alternatives = (alt for alt in alternatives if os.path.exists (alt))
            try:
                f = next (iter (alternatives))
            except StopIteration as e:
                logging.warning (f'Missing file {f}')
                continue

        filepaths.append (f)
    return filepaths
#
# Transform wireless files to concatenated binary, single electrode, files
#
def wirelessToBinV2(inDir, outDir, files, elecList,
                    prefix=['NEUR'], suffix='DF1',
                    nchannels=DATANCHANNELS, freq=DATAFREQ, 
                    verbose=False, savetype=np.int16):
    """
    Read data files and return the neuronal channels data, optionally save them in
    files. Data is saved as numpy binary data. (X.tofile)

    Parameters:
    inDir - Directroy where the files reside (None only uses file path in files).
    outDir - Output files directory. If None, no files are saved.
    files - A list of file names or file numbers.
    eleList - A list of electrodes to save (0-nchannels).
    prefix - List of filename prefix. When files are provided as numbers, the prefix
            list is used to search for the file in inDir. First matched file is
            used as the file. Default ['NEUR'].
    suffix - Data files suffix. Used to construct filename when files are provided
            as numbers. Default 'DF1'.
    nchannels - Number of neuronal channels recorded. Default is DATANCHANNELS (32).
    freq - Sampling frequency. Default is DATAFREQ (32KHz).
    verbose - Whether to log more information.

    Returns:
    channels, timestamps
    Channels [nsamples x nchannels] is the neuronal data. All channels are returned.
    Timestamps [nblocks x 2] are the beginning and finish timestamps of each block.
    """
    #
    # Get input file list.
    #
    filepaths = getDataFiles (inDir, files, prefix, suffix, verbose)

    channels, timestamps = wirelessToChannels (None, filepaths, verbose=verbose,
                                               nchannels=nchannels, freq=freq,
                                               savetype=savetype)
    elecfiles = None
    if not (outDir is None):
        safeOutputDir (outDir)
        #
        # Find first and last file numbers.
        #
        nums = (int (re.sub ('\D', '', os.path.basename (f).split ('.') [0]))
                for f in filepaths)
        nums = sorted (nums)
        file0, filen = nums [0], nums [-1]
        #
        # Create electrode files.
        #
        elecfilenames = (os.path.join (outDir, f'Elec{e}-F{file0}T{filen}.bin')
                         for e in elecList)
        for elec, name in zip (elecList, elecfilenames):
            channels [:, elec].tofile (open (name, 'wb'))
            if verbose:
                logging.info (f'Electrode {elec} dumped to file {name}.')

    return channels, np.linspace (0,
                                  (timestamps [-1][1] - timestamps [0][0])/1000,
                                  len (channels))
#
# Transform wireless data to motion data
#
def wirelessToMotion (base, files, outdir=None, prefix='NEUR',
                      verbose=False, tolerance=2, blocktolerance=3):
    logging.info("started wirelessToMotion function")
    sensors = ['acc', 'gyr', 'mag']
    axes = ['x', 'y', 'z']
    files = (f if type (f) is str else f'{prefix}{f:04d}.DT2'
             for f in files)
    if not base is None:
        files = (os.path.join (base, datafile) for datafile in files)

    timestamps = []
    datawords = []
    offsets = []
    spans = []
    goodblocks = []
    # Original data for each of the data types.
    origdata = [[], [], []]
    prevtimestamp = 0
    for datafile in files:
        if verbose:
            print(f'Read raw file {datafile}')

        fd = open(datafile, 'rb')
        data = np.fromfile(fd, dtype=np.uint16)
        fd.close()

        # Data files should be fixed length.
        if len (data)*2 != DATAFILELEN:
            raise Exception (f'File {datafile} size is {len (data)*2}!')

        # Data shape is n x channels
        data = data.reshape (-1, DATANCHANNELS)
        #
        # Motion data is in blocks of 1024 words.
        # As motion data is a channel in the data, each block neuronal data is
        # 1024*32KHz = 32mSec. However for motion data we have timestamps. In
        # cases where we don't have a timestamp (bad block) we assume 32mSec.
        #
        motiondata = data [:,DMCHANNEL].reshape (-1, 1024)

        cgoodblocks = (
            np.all (motiondata [:, :2] == DMSIGNATURE, axis=1) &
            (motiondata [:, 9] == 0)
                     )

        cbadblocks = np.nonzero (np.logical_not (cgoodblocks)) [0]
        if not np.any (cgoodblocks):
            # No good blocks in file. Add zeros to data.
            timestamps.append (np.zeros_like (cgoodblocks))
            datawords.append (np.zeros_like (cgoodblocks))
            offsets.append (np.zeros_like (cgoodblocks))
            spans.append (np.ones_like (cgoodblocks)*32)
            goodblocks.append (cgoodblocks)
            origdata.append (np.zeros ((1, 9)))

        ctimestamps = np.array (motiondata [:, 10] + motiondata [:, 11] * 2**16)
        cdatawords  = motiondata [:, 6:9]
        coffsets    = motiondata [:, 2:5]
        cspans      = np.zeros_like (ctimestamps)
        cspans [:-1] = ctimestamps [1:] - ctimestamps [:-1]
        # Sanity check on timestamps
        insaneblocks = (cspans > 2048) | (cspans < 0)
        cgoodblocks [insaneblocks] = False
        cbadblocks = np.nonzero (np.logical_not (cgoodblocks)) [0]

        if len (cbadblocks) > 0:
            start, finish, length = runs_of_ones_array (
                np.logical_not (cgoodblocks))
            logging.warning (f'{datafile}: Bad blocks at: ' +
                             ' '.join ([f'{s}-{f-1} ({l})' for s, f, l in
                                        zip (start, finish, length)]))

        # logging.info (f'CSpans {cspans}')
        # logging.info (f'CTimestamps {ctimestamps}')
        # Fix last block of previous file using first timestamp.
        if len (spans) > 0:
            spans [-1] [-1] = ctimestamps [0] - timestamps [-1] [-1]
        else:
            # Handle first file. First block is empty.
            cspans [0] = 0
        #
        # Handle badblocks:
        # Block timespan
        # --------------
        # Since we have no indication of the block's timestamp we need to make
        # assumptions. Since a single block is 1024 words @ 32KHz, each block
        # spans 32mSec of neuronal data.
        #
        # Values
        # ------
        # Insert 0 value for the appropriate length. Since the block is bad we
        # just overwrite the block with zeros and reference them. Although
        # magnetormeter data is usually sampled at a lower frequency, we're
        # resampling the data anyhow, so it's not importanct to follow the
        # sample rate.
        #
        cspans     [cbadblocks] = 32 * 16
        motiondata [cbadblocks,
                    DMHEADERSIZE:(DMHEADERSIZE+32*3*3)] = 0
        cdatawords [cbadblocks] = [32*3, 32*3, 32*3]
        coffsets [cbadblocks] = (DMHEADERSIZE + 32*3*0,
                                 DMHEADERSIZE + 32*3*1,
                                 DMHEADERSIZE + 32*3*2)
        #
        # Add data to processes files lists.
        #
        timestamps.append (ctimestamps)
        datawords.append  (cdatawords)
        offsets.append    (coffsets)
        spans.append      (cspans)
        goodblocks.append (cgoodblocks)
        origdata [0].append (np.concatenate (
            [motiondata [block, offset:offset+datalen].astype (np.int16)
             for block, (offset, datalen)
             in enumerate (zip (coffsets [:,0], cdatawords [:,0]))]
        ))
        origdata [1].append (np.concatenate (
            [motiondata [block, offset:offset+datalen].astype (np.int16)
             for block, (offset, datalen)
             in enumerate (zip (coffsets [:,1], cdatawords [:,1]))]
        ))
        origdata [2].append (np.concatenate (
            [motiondata [block, offset:offset+datalen].astype (np.int16)
             for block, (offset, datalen)
             in enumerate (zip (coffsets [:,2], cdatawords [:,2]))]
        ))

        # print ([(block, offset, datalen, motiondata [block].shape)
        #      for block, (offset, datalen)
        #      in enumerate (zip (coffsets [0], cdatawords [0]))])
        prevtimestamp = ctimestamps [-1]

    # Flatten data from all files and remove trailing empty blocks
    goodblocks = np.concatenate (goodblocks)
    if not np.any (goodblocks):
        logging.error (f'No good blocks were found in files.')
        resampled = np.zeros ((0, 3*3))
        df = pd.DataFrame (resampled,
                           columns=pd.MultiIndex
                           .from_product ([sensors, axes], 
                                          names=['sensorName', 'sensorNum']))
        return df
        
    # We drop the last block so we have valid timestamps.
    lastgood = np.where (goodblocks) [0] [-1]
    goodblocks = goodblocks [:lastgood]
    origdata = [np.concatenate (data).reshape (-1, 3) for data in origdata]
    timestamps = np.concatenate (timestamps) [:lastgood+1]
    timestamps = timestamps / DMSAMPLERATE / DMTSFACTOR
    datawords  = np.concatenate (datawords) [:lastgood]
    offsets    = np.concatenate (offsets) [:lastgood]
    spans      = (np.concatenate (spans) / DMTSFACTOR) [:lastgood].astype (int)

    # Fix last span. As we don't have the next timestamp, we assume 32mSec.
    spans [-1] = 32
    # Resample data if necessary. Mismatched is 0 for correct # of samples.
    resampled = np.zeros ((int (np.sum (spans)), 3*3))
    mismatched = (spans.reshape (-1, 1) - datawords [:lastgood]/3)
    intolerable = np.abs (mismatched) > tolerance
    btimes   = np.concatenate (([0], np.add.accumulate (spans))).astype (int)
    # logging.info (f'BTimes {btimes}')
    logging.info (f'Resampling')
    for datatype in range (2):
        # Copy non mismatched good blocks
        logging.info (f'Copying good blocks {datatype}.')
        boffsets = np.concatenate (([0],
                   np.add.accumulate (datawords [:, datatype]/3))).astype (int)
        s, f, n = runs_of_ones_array ((mismatched [:, datatype] == 0) * 1)
        for bstart, bfinish, nblocks in zip (s, f, n):
            target0 = btimes [bstart]
            target1 = btimes [bfinish]
            source0 = boffsets [bstart]
            source1 = boffsets [bfinish]

            resampled [target0:target1, datatype*3:(datatype+1)*3] = (
                origdata [datatype] [source0:source1]
            )
        # Fix over tolerance blocks.
        logging.info (f'Fixing intolerable blocks {datatype}.')
        for b,e,l in zip (*runs_of_ones_array (intolerable [:, datatype])):
            # This is a reversed graph with X as the position of samples
            # (integer values are samples), and Y the timestampes. Since we need
            # to interpolate on a different timescale we compute the timestamp
            # of each sampled point within a block by interpolation of Y, and
            # setting the expected number of samples within the block (instead
            # of the existing one).
            # Generate resampled x points in the run
            x = np.arange (np.sum (datawords [b:e, datatype] / 3))
            # Compute the location of sampled timestamps
            xp = np.add.accumulate (
                np.concatenate (([0], datawords [b:e, datatype] / 3))
                )
            # The timestamps at block boundaries
            yp = timestamps [b:e+1] * DMSAMPLERATE
            source0 = boffsets [b]
            source1 = boffsets [e]
            target0 = btimes [b]
            target1 = btimes [e]
            # Use interpolate to locate the sample timestamps within blocks.
            st = np.interp (x, xp, yp)
            # Reset start to 0
            st = (st - st [0])
            # Resample the run
            logging.info (f'Resampling run {b}-{e}:{l}')
            logging.info (f's0 {source0}:s1 {source1}, t0 {target0}:t1 {target1}')
            for i, ri in zip (range (3), range (datatype*3, datatype*3+3)):
                resampled [target0:target1, ri] = np.interp (
                    np.arange (target1 - target0),
                    st, origdata [datatype] [source0:source1, i])

        # Set fixed blocks as good.
        # Fix blocks within tolerance where aberration persists.
        summed = np.add.accumulate (mismatched [:, datatype])
        # Compute over tolerance blocks
        state = summed [0]
        fixedpos = 0
        dataoff = datatype*3
        logging.info (f'Fixing long aberration {datatype}.')
        for i in np.where (summed [:-1] != summed [1:]) [0]:
            # Have we handled this transition already
            if i < fixedpos or intolerable [i+1, datatype]:
#                print (f'Skipping fixed {fixedpos} or intolerable block {i+1}')
                state = summed [i+1]
                continue

            # Does this aberration right itself.
            pos0 = np.where (summed [i+2:i+1+blocktolerance] == state) [0]
#            print (i+1, pos0)
            if len (pos0) > 0:
                # We have a 0 - This block rights itself in the next blocks.
                pos0 = pos0 [0] + i+2
#                print (f'Self righting block {i+1} @ {pos0}:  {source0}:{source1} -> {target0}:{target1}/{state}')
                # Check no intolerable blocks interrupt.
                if not np.any (intolerable [pos0:i+1+blocktolerance, datatype]):
                    source0 = boffsets [i+1]
                    source1 = boffsets [pos0+1]
                    target0 = btimes [i+1]
                    target1 = btimes [pos0+1]
                    resampled [target0:target1, dataoff:dataoff+3] = (
                        origdata [datatype] [source0:source1]
                        )
                    fixedpos = pos0
                    continue

            # We need to resample the block.
            #x = np.arange (np.sum (datawords [b:e, datatype] / 3))
            # Compute the location of sampled timestamps
            #xp = np.array ([0, datawords [i+1] / 3)
            # The timestamps at block boundaries
            #yp = timestamps [i+1:i+3] * DMSAMPLERATE
            source0 = boffsets [i+1]
            source1 = boffsets [i+2]
            target0 = btimes [i+1]
            target1 = btimes [i+2]
            # Use interpolate to locate the sample timestamps within blocks.
            #st = np.interp (x, xp, yp)
            # Reset start to 0
            #st = (st - st [0])
            st = np.linspace (0,
                              timestamps [i+2]-timestamps [i+1],
                              source1-source0+1) [:-1] * DMSAMPLERATE
            # Resample the run
            for si, ti in zip (range (3), range (datatype*3, datatype*3+3)):
#                print ('Data')
#                print (origdata [datatype] [source0:source1, i])
#                print ('Sample times')
#                print (st)
#                print (f'New range: {target1 - target0}')
                resampled [target0:target1, ti] = np.interp (
                    np.arange (target1 - target0),
                    st, origdata [datatype] [source0:source1, si])
#            print (f'Resampled block {source0}:{source1} -> {target0}:{target1}/{state} {i}{summed [i-5:i+5]}')

            state = summed [i+1]
            fixedpos = i+1

    # Handle the magnetometer data using resample. Since it's sampled at 111Hz,
    # we'll resample every 9 blocks taking 7 blocks each time.
    boffsets = np.concatenate (([0],
                  np.add.accumulate (datawords [:, 2]/3))).astype (int)
    logging.info (f'Resampling magnetometer.')
    for b, e, l in zip (*runs_of_ones_array (goodblocks [:-1])):
        logging.info (f'Run {b}-{e}:{l}')
        for si, ti in zip (range (3), range (6, 9)):
            if l < 9:
                source0 = boffsets [b]
                source1 = boffsets [e]
                target0 = btimes [b]
                target1 = btimes [e]
                # For short sequences we have to resample as is
                resampled [target0:target1, ti] = (
                    sig.resample (origdata [2] [source0:source1, si],
                                  target1 - target0)
                    )
                continue

            # First block has a boundary issue
            source0 = boffsets [b]
            source1 = boffsets [b+9]
            target0 = btimes [b]
            target1 = btimes [b+8]
            target2 = btimes [b+9]
            resampled [target0:target1, ti] = (
                sig.resample (origdata [2] [source0:source1, si],
                              target2 - target0) [:target1 - target0]
                )
            # Handle all interim resamples (available
            data = []
            # for bi in range (b+8, e-8, 7):
            #     try:
            #         data.append (sig.resample (origdata [2] [boffsets [bi]:boffsets [bi+9],
            #                                         si],
            #                           btimes [bi+8] - btimes [bi-1])
            #             [btimes [bi]-btimes [bi-1]:btimes [bi+7] - btimes [bi-1]])
            #     except Exception as e:
            #         print (f'Exception @{bi}')
            #         print (btimes [bi-1], btimes [bi+8])
            #         print (boffsets [bi], boffsets [bi+9])
            #         traceback.print_exception (type (ex), ex, ex.__traceback__)
            data = [sig.resample (origdata [2] [boffsets [bi]:boffsets [bi+9],
                                                si],
                                  btimes [bi+8] - btimes [bi-1])
                    [btimes [bi]-btimes [bi-1]:btimes [bi+7] - btimes [bi-1]]
                    for bi in range (b+8, e-8, 7)]
            data = np.concatenate (data)
            target0 = target1
            target1 = target0 + len (data)
            resampled [target0:target1, ti] = data
            # Handle last resample block
            nb = b+8 + (e - b-8) // 7 * 7
            source0 = boffsets [nb - 1]
            source1 = boffsets [e+1]
            targetb = btimes [nb-1]
            target0 = btimes [nb]
            target1 = btimes [e+1]
            resampled [target0:target1, ti] = (
                sig.resample (origdata [2] [source0:source1, si],
                              target1 - targetb) [target0-targetb:]
                )
            
    df = pd.DataFrame (resampled,
                       columns=pd.MultiIndex
                       .from_product ([sensors, axes], 
                                      names=['sensorName', 'sensorNum']),)
    return df

#
# Transform wireless data to motion data
#
def wirelessToMotionV2 (base, files, prefix=[ 'NEUR' ], suffix='DF1',
                        verbose=False, tolerance=2, blocktolerance=3):
    logging.info("started wirelessToMotion function")
    sensors = ['acc', 'gyr', 'mag']
    axes = ['x', 'y', 'z']
    files = getDataFiles (base, files, prefix, suffix, verbose)

    btimestamps = []   # These are the block timestamps. They're mostly ignored.
    timestamps = []
    datawords = []
    offsets = []
    spans = []
    goodblocks = []
    # Original data for each of the data types.
    origdata = [[], [], []]
    prevtimestamp = 0

    for datafile in files:
        if verbose:
            print(f'Read raw file {datafile}')

        fd = open(datafile, 'rb')
        data = np.fromfile(fd, dtype=np.uint16)
        fd.close()

        #
        # Generic Stage
        #
        # Data files should be fixed length.
        if len (data)*2 != DATAFILELEN:
            raise Exception (f'File {datafile} size is {len (data)*2}!')

        # File is composed of 64KB blocks.
        blocks = data.reshape (-1, BLOCKLEN // 2)
        # Test blocks signatures.
        filegoodblocks = np.all (
            blocks [:, :len(BLOCKSIGNATURE)] == BLOCKSIGNATURE, axis=1)
        # Extract timestamps.
        #timestamps = np.dot (blocks [:, 8:10], [[1], [2**16]])
        #timestamps = timestamps.astype (np.uint32).reshape (-1)
        # Extract data partitions.
        # Partition info are 3 uint32: Type, Start, Length.
        nblocks = len (blocks)
        compmat = (np.diag (np.ones (7*6)) [::2] +
                   np.diag (np.ones (7*6)) [1::2] * 2**16).T
        partinfo = np.dot (blocks [:, 12:54], compmat).reshape (nblocks, -1, 3)
        # Get reference to neuronal data.
        ind0, ind1 = np.where (partinfo [filegoodblocks, :, 0] == MOTIONTYPE)
        if np.any (np.unique (ind0, return_counts=True) [1] > 1):
            raise f'Non unique neuronal data partition in {datafile}'

        motiondata = []
        for i0, i1 in zip (ind0, ind1):
            _, start, length = partinfo [i0, i1]
            motiondata.append (
                blocks [i0, int (start // 2):int ((start + length) // 2)]
            )
        motionheads = np.array ([d [:16] for d in motiondata])
        cgoodblocks = (
            np.all (motionheads [:, :2] == DMSIGNATURE, axis=1) &
            (motionheads [:, 9] == 0)
                     )
#        print (f'{datafile}: Blocks shape {blocks.shape} Good {cgoodblocks}')

        cbadblocks = np.nonzero (np.logical_not (cgoodblocks)) [0]
        if not np.any (cgoodblocks):
            # No good blocks in file. Add zeros to data.
            timestamps.append (np.zeros_like (cgoodblocks))
            datawords.append (np.zeros_like (cgoodblocks))
            offsets.append (np.zeros_like (cgoodblocks))
            spans.append (np.ones_like (cgoodblocks)*32)
            goodblocks.append (cgoodblocks)
            origdata.append (np.zeros ((1, 9)))

        ctimestamps = np.array (motionheads [:, 10] +
                                motionheads [:, 11] * 2**16)
        ctimestamps2 = np.array ((motionheads [:, 10],
                                  motionheads [:, 11]))
        cdatawords  = motionheads [:, 6:9]
        coffsets    = motionheads [:, 2:5]
        cspans      = np.zeros_like (ctimestamps)
        cspans [:-1] = ctimestamps [1:] - ctimestamps [:-1]
        # Sanity check on timestamps
        insaneblocks = (cspans > 2048) | (cspans < 0)
        cgoodblocks [insaneblocks] = False
        cbadblocks = np.nonzero (np.logical_not (cgoodblocks)) [0]

#        print (f'{datafile}: Insane blocks {runs_of_ones_array (insaneblocks) [0]}')
#        print (f'{datafile}: Good {runs_of_ones_array (cgoodblocks) [0]}')
#        print (f'{datafile}: Bad {runs_of_ones_array (cbadblocks) [0]}')
#        print (f'{datafile}: cspans {cspans}')
#        print (f'{datafile}: timestamps {ctimestamps}')
#        print (f'{datafile}: timestamps2 {ctimestamps2}')
#        print (f'{datafile}: words {[cdatawords [i] for i in np.where (insaneblocks) [0]]}')

        if len (cbadblocks) > 0:
            start, finish, length = runs_of_ones_array (
                np.logical_not (cgoodblocks))
            logging.warning (f'{datafile}: Bad blocks at: ' +
                             ' '.join ([f'{s}-{f-1} ({l})' for s, f, l in
                                        zip (start, finish, length)]))

        # logging.info (f'CSpans {cspans}')
        # logging.info (f'CTimestamps {ctimestamps}')
        # Fix last block of previous file using first timestamp.
        if len (spans) > 0:
            spans [-1] [-1] = ctimestamps [0] - timestamps [-1] [-1]
        else:
            # Handle first file. First block is empty.
            cspans [0] = 0
        #
        # Handle badblocks:
        # Block timespan
        # --------------
        # Since we have no indication of the block's timestamp we need to make
        # assumptions. Since a single block is 1024 words @ 32KHz, each block
        # spans 32mSec of neuronal data.
        #
        # Values
        # ------
        # Insert 0 value for the appropriate length. Since the block is bad we
        # just overwrite the block with zeros and reference them. Although
        # magnetormeter data is usually sampled at a lower frequency, we're
        # resampling the data anyhow, so it's not importanct to follow the
        # sample rate.
        #
        cspans     [cbadblocks] = 32 * 16
        for block in cbadblocks:
            motiondata [block] [DMHEADERSIZE:(DMHEADERSIZE+32*3*3)] = 0
        cdatawords [cbadblocks] = [32*3, 32*3, 32*3]
        coffsets [cbadblocks] = (DMHEADERSIZE + 32*3*0,
                                 DMHEADERSIZE + 32*3*1,
                                 DMHEADERSIZE + 32*3*2)
        #
        # Add data to processes files lists.
        #
        timestamps.append (ctimestamps)
        datawords.append  (cdatawords)
        offsets.append    (coffsets)
        spans.append      (cspans)
        goodblocks.append (cgoodblocks)
        origdata [0].append (np.concatenate (
            [motiondata [block] [offset:offset+datalen].astype (np.int16)
             for block, (offset, datalen)
             in enumerate (zip (coffsets [:,0], cdatawords [:,0]))]
        ))
        origdata [1].append (np.concatenate (
            [motiondata [block] [offset:offset+datalen].astype (np.int16)
             for block, (offset, datalen)
             in enumerate (zip (coffsets [:,1], cdatawords [:,1]))]
        ))
        origdata [2].append (np.concatenate (
            [motiondata [block] [offset:offset+datalen].astype (np.int16)
             for block, (offset, datalen)
             in enumerate (zip (coffsets [:,2], cdatawords [:,2]))]
        ))

        # print ([(block, offset, datalen, motiondata [block].shape)
        #      for block, (offset, datalen)
        #      in enumerate (zip (coffsets [0], cdatawords [0]))])
        prevtimestamp = ctimestamps [-1]

    # Flatten data from all files and remove trailing empty blocks
    goodblocks = np.concatenate (goodblocks)
    if not np.any (goodblocks):
        logging.error (f'No good blocks were found in files.')
        resampled = np.zeros ((0, 3*3))
        df = pd.DataFrame (resampled,
                           columns=pd.MultiIndex
                           .from_product ([sensors, axes], 
                                          names=['sensorName', 'sensorNum']))
        return df
        
    # We drop the last block so we have valid timestamps.
    lastgood = np.where (goodblocks) [0] [-1]
    goodblocks = goodblocks [:lastgood]
    origdata = [np.concatenate (data).reshape (-1, 3) for data in origdata]
    timestamps = np.concatenate (timestamps) [:lastgood+1]
    timestamps = timestamps / DMSAMPLERATE / DMTSFACTOR
    datawords  = np.concatenate (datawords) [:lastgood]
    offsets    = np.concatenate (offsets) [:lastgood]
    spans      = (np.concatenate (spans) / DMTSFACTOR) [:lastgood].astype (int)

    # Fix last span. As we don't have the next timestamp, we assume 32mSec.
    spans [-1] = 32
    # Resample data if necessary. Mismatched is 0 for correct # of samples.
    resampled = np.zeros ((int (np.sum (spans)), 3*3))
    mismatched = (spans.reshape (-1, 1) - datawords [:lastgood]/3)
    intolerable = np.abs (mismatched) > tolerance
    btimes   = np.concatenate (([0], np.add.accumulate (spans))).astype (int)
    # logging.info (f'BTimes {btimes}')
    logging.info (f'Resampling')
    for datatype in range (2):
        # Copy non mismatched good blocks
        logging.info (f'Copying good blocks {datatype}.')
        boffsets = np.concatenate (([0],
                   np.add.accumulate (datawords [:, datatype]/3))).astype (int)
        s, f, n = runs_of_ones_array ((mismatched [:, datatype] == 0) * 1)
        for bstart, bfinish, nblocks in zip (s, f, n):
            target0 = btimes [bstart]
            target1 = btimes [bfinish]
            source0 = boffsets [bstart]
            source1 = boffsets [bfinish]

            resampled [target0:target1, datatype*3:(datatype+1)*3] = (
                origdata [datatype] [source0:source1]
            )
        # Fix over tolerance blocks.
        logging.info (f'Fixing intolerable blocks {datatype}.')
        for b,e,l in zip (*runs_of_ones_array (intolerable [:, datatype])):
            # This is a reversed graph with X as the position of samples
            # (integer values are samples), and Y the timestampes. Since we need
            # to interpolate on a different timescale we compute the timestamp
            # of each sampled point within a block by interpolation of Y, and
            # setting the expected number of samples within the block (instead
            # of the existing one).
            # Generate resampled x points in the run
            x = np.arange (np.sum (datawords [b:e, datatype] / 3))
            # Compute the location of sampled timestamps
            xp = np.add.accumulate (
                np.concatenate (([0], datawords [b:e, datatype] / 3))
                )
            # The timestamps at block boundaries
            yp = timestamps [b:e+1] * DMSAMPLERATE
            source0 = boffsets [b]
            source1 = boffsets [e]
            target0 = btimes [b]
            target1 = btimes [e]
            # Use interpolate to locate the sample timestamps within blocks.
            st = np.interp (x, xp, yp)
            # Reset start to 0
            st = (st - st [0])
            # Resample the run
            logging.info (f'Resampling run {b}-{e}:{l}')
            logging.info (f's0 {source0}:s1 {source1}, t0 {target0}:t1 {target1}')
            for i, ri in zip (range (3), range (datatype*3, datatype*3+3)):
                resampled [target0:target1, ri] = np.interp (
                    np.arange (target1 - target0),
                    st, origdata [datatype] [source0:source1, i])

        # Set fixed blocks as good.
        # Fix blocks within tolerance where aberration persists.
        summed = np.add.accumulate (mismatched [:, datatype])
        # Compute over tolerance blocks
        state = summed [0]
        fixedpos = 0
        dataoff = datatype*3
        logging.info (f'Fixing long aberration {datatype}.')
        for i in np.where (summed [:-1] != summed [1:]) [0]:
            # Have we handled this transition already
            if i < fixedpos or intolerable [i+1, datatype]:
#                print (f'Skipping fixed {fixedpos} or intolerable block {i+1}')
                state = summed [i+1]
                continue

            # Does this aberration right itself.
            pos0 = np.where (summed [i+2:i+1+blocktolerance] == state) [0]
#            print (i+1, pos0)
            if len (pos0) > 0:
                # We have a 0 - This block rights itself in the next blocks.
                pos0 = pos0 [0] + i+2
#                print (f'Self righting block {i+1} @ {pos0}:  {source0}:{source1} -> {target0}:{target1}/{state}')
                # Check no intolerable blocks interrupt.
                if not np.any (intolerable [pos0:i+1+blocktolerance, datatype]):
                    source0 = boffsets [i+1]
                    source1 = boffsets [pos0+1]
                    target0 = btimes [i+1]
                    target1 = btimes [pos0+1]
                    resampled [target0:target1, dataoff:dataoff+3] = (
                        origdata [datatype] [source0:source1]
                        )
                    fixedpos = pos0
                    continue

            # We need to resample the block.
            #x = np.arange (np.sum (datawords [b:e, datatype] / 3))
            # Compute the location of sampled timestamps
            #xp = np.array ([0, datawords [i+1] / 3)
            # The timestamps at block boundaries
            #yp = timestamps [i+1:i+3] * DMSAMPLERATE
            source0 = boffsets [i+1]
            source1 = boffsets [i+2]
            target0 = btimes [i+1]
            target1 = btimes [i+2]
            # Use interpolate to locate the sample timestamps within blocks.
            #st = np.interp (x, xp, yp)
            # Reset start to 0
            #st = (st - st [0])
            st = np.linspace (0,
                              timestamps [i+2]-timestamps [i+1],
                              source1-source0+1) [:-1] * DMSAMPLERATE
            # Resample the run
            for si, ti in zip (range (3), range (datatype*3, datatype*3+3)):
#                print ('Data')
#                print (origdata [datatype] [source0:source1, i])
#                print ('Sample times')
#                print (st)
#                print (f'New range: {target1 - target0}')
                resampled [target0:target1, ti] = np.interp (
                    np.arange (target1 - target0),
                    st, origdata [datatype] [source0:source1, si])
#            print (f'Resampled block {source0}:{source1} -> {target0}:{target1}/{state} {i}{summed [i-5:i+5]}')

            state = summed [i+1]
            fixedpos = i+1

    # Handle the magnetometer data using resample. Since it's sampled at 111Hz,
    # we'll resample every 9 blocks taking 7 blocks each time.
    boffsets = np.concatenate (([0],
                  np.add.accumulate (datawords [:, 2]/3))).astype (int)
    logging.info (f'Resampling magnetometer.')
    for b, e, l in zip (*runs_of_ones_array (goodblocks [:-1])):
        logging.info (f'Run {b}-{e}:{l}')
        for si, ti in zip (range (3), range (6, 9)):
            if l < 9:
                source0 = boffsets [b]
                source1 = boffsets [e]
                target0 = btimes [b]
                target1 = btimes [e]
                # For short sequences we have to resample as is
                resampled [target0:target1, ti] = (
                    sig.resample (origdata [2] [source0:source1, si],
                                  target1 - target0)
                    )
                continue

            # First block has a boundary issue
            source0 = boffsets [b]
            source1 = boffsets [b+9]
            target0 = btimes [b]
            target1 = btimes [b+8]
            target2 = btimes [b+9]
            resampled [target0:target1, ti] = (
                sig.resample (origdata [2] [source0:source1, si],
                              target2 - target0) [:target1 - target0]
                )
            # Handle all interim resamples (available
            data = []
            # for bi in range (b+8, e-8, 7):
            #     try:
            #         data.append (sig.resample (origdata [2] [boffsets [bi]:boffsets [bi+9],
            #                                         si],
            #                           btimes [bi+8] - btimes [bi-1])
            #             [btimes [bi]-btimes [bi-1]:btimes [bi+7] - btimes [bi-1]])
            #     except Exception as e:
            #         print (f'Exception @{bi}')
            #         print (btimes [bi-1], btimes [bi+8])
            #         print (boffsets [bi], boffsets [bi+9])
            #         traceback.print_exception (type (ex), ex, ex.__traceback__)
            data = [sig.resample (origdata [2] [boffsets [bi]:boffsets [bi+9],
                                                si],
                                  btimes [bi+8] - btimes [bi-1])
                    [btimes [bi]-btimes [bi-1]:btimes [bi+7] - btimes [bi-1]]
                    for bi in range (b+8, e-8, 7)]
            data = np.concatenate (data)
            target0 = target1
            target1 = target0 + len (data)
            resampled [target0:target1, ti] = data
            # Handle last resample block
            nb = b+8 + (e - b-8) // 7 * 7
            source0 = boffsets [nb - 1]
            source1 = boffsets [e+1]
            targetb = btimes [nb-1]
            target0 = btimes [nb]
            target1 = btimes [e+1]
            resampled [target0:target1, ti] = (
                sig.resample (origdata [2] [source0:source1, si],
                              target1 - targetb) [target0-targetb:]
                )
            
    df = pd.DataFrame (resampled,
                       columns=pd.MultiIndex
                       .from_product ([sensors, axes], 
                                      names=['sensorName', 'sensorNum']),)
    return df

def wirelessToMotioncV0(inDir, files, outDir=None, filepfx='NEUR', verbose=False, samplingRate=32000):
    logging.info("started wirelessToMotion function")
    motionData = np.empty(shape=0)
    files = [f if type (f) is str else f'{filepfx}{f:04d}.DT2' for f in files]
    # Read each wireless file 
    for fileName in files:
        if not inDir is None:
            fileName = os.path.join (inDir, fileName)

        if verbose:
            print(f'Read raw file {fileName}')

        fid = open(fileName, 'rb')
        fileData = np.fromfile(fid, dtype=np.int16)
        motionData = np.concatenate((motionData, fileData[0::32]))
        fid.close()

    bBlock = range(0, motionData.shape[0], 1024)
    nBlock = len(bBlock)
    logging.info(f'Total data length: {motionData.shape[0]} Number of blocks {nBlock}')
    blockIndex = {"acc": (2, 6, np.zeros((nBlock, 2), dtype=np.int32)),
                  "gyr": (3, 7, np.zeros((nBlock, 2), dtype=np.int32)),
                  "mag": (4, 8, np.zeros((nBlock, 2), dtype=np.int32))}
    for i, block in enumerate(bBlock):
        if motionData[block] != 13579 or motionData[block + 1] != 24680:
            logging.warning(f'Error in block {i} - Block starts with ({int(motionData[block])},{int(motionData[block + 1])}) instead of (13579,24680)')
        else:
            for index in blockIndex:
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
                logging.warning(f'Resample {sensor} {axis} - acquisition data({msElecData}ms) != motion({msMotionData}ms)')
                kinematics[sensor][axis] = sig.resample(kinematics[sensor][axis], msElecData).astype(np.int16)
            df[sensor, axis] = kinematics[sensor][axis]

    if outDir is not None:
        safeOutputDir(outDir)
        fileName = "{0}Motion-F{1}T{2}.pkl".format(outDir, files[0], files[-1])
        logging.info(f'Writing output to file {fileName}')
        with open(fileName, 'wb') as file:
            pickle.dump(df, file)

    return df

#
# Transform wireless data to motion data
#
def binToLFP(inDir, outDir, filePattern, elecList, freq=[2, 300], notch=False, verbose=False, savetype=np.int16):
    logging.info("started binToLFP function")
    safeOutputDir(outDir)
    [bf, af] = sig.butter(3, [f / (1000 / 2) for f in freq], btype='band')
    for elec in elecList:
        inFileName = filePattern.format(inDir, str(elec))
        with open(inFileName, 'rb') as ifid:
            data = np.fromfile(ifid, dtype=np.int16)
        sdata = sig.resample(data, num=data.shape[0] // 32)
        sdata = sig.filtfilt(bf, af, sdata).astype(savetype)
        if notch:
            F0, Q, Fs = 50, 35, 1000
            [bcomb, acomb] = sig.iirnotch(F0, Q, Fs)
            sdata = sig.filtfilt(bcomb, acomb, sdata).astype(savetype)
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
    offset = 32000 * 90
    count = 32000 * 240
    samplingRate = 32000

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
        fig, ax = plt.subplots(figsize=(8, 4), nrows=1, ncols=3, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4), nrows=1, ncols=2, sharex=True)

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
                  raw_fold='binNew', st=40):
    fig, ax = plt.subplots(figsize=(8, len(elecList) * 2), nrows=len(elecList) * 2, ncols=1, sharex=True, sharey=True)
    try:
        plt.ylim(ylim)
    except:
        print(f'y axis not limited, possibly wrong input')
    samplingRate = 32000
    bs = st*samplingRate
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
    fig.tight_layout()

def trimoscillations (data, clip=30000, hp=None, lp=None, fs=1000):
    """
    Trim accelerations erroneous oscillations, occuring as a result of
    bumping the device. Two passes are done:
    1. Remove oscillations of +/- limit and replace with interpolation.
    2. Use a LPF on the data.

    Parameters:
    data - The kinematic data as dataframe.
    clip - The limits to clip in the multiple oscillations (None skips).
    filtfreq - The filter frequency (None skips).

    Returns:
    The data trimmed.
    """
    
    SENSORS = [ 'acc', 'gyr', 'mag' ]
    AXES = [ 'x', 'y', 'z' ]

    data = data.astype (float)
    if clip != None:
        dims = [(sensor, ax) for sensor in SENSORS for ax in AXES]
        smeardims = [(sensor, ax) for sensor in SENSORS for ax in AXES]
        smearindex = [dim in smeardims for dim in dims]
        #
        # Find values crossing the clip window.
        #
        extremes = (
            ((data.iloc [:-1] [dims] >  clip).to_numpy () &
             (data.iloc [1:]  [dims] < -clip).to_numpy ()) |
            ((data.iloc [:-1] [dims] < -clip).to_numpy () &
             (data.iloc [1:]  [dims] >  clip).to_numpy ())
        )
        #
        # Smear the extremes right and left to take care of the boundaries.
        #
        extremes [:-1, smearindex] |= extremes [1:, smearindex]
        extremes [:-2, smearindex] |= extremes [2:, smearindex]
        extremes [4:, smearindex] |= extremes [:-4, smearindex]
        #
        # For each axis find the regions and interpolate.
        #
        for iax, (sensor, ax) in enumerate (dims):
            # Locate start and end of regions.
            bregions = np.where ((extremes [:-1, iax] == False) &
                                 (extremes [1:, iax] == True)) [0]
            eregions = np.where ((extremes [:-1, iax] == True) &
                                 (extremes [1:, iax] == False)) [0]
            # Ensure all regions have a beginning and an end.
            if extremes [0, iax]:
                bregions = np.insert (bregions, 0, [0])
            if extremes [-1, iax]:
                eregions = np.append (eregions, extremes.shape [0]+1)
            # Iterate over regions and interpolate
            for btime, etime in zip (bregions, eregions):
                btime = int (btime) + 1
                etime = int (etime) + 1

                # if self.verbose >= 1:
                #     logging.info (f'Region axis: {sensor} {ax} {btime}:{etime}')

                begin = data.index [btime]
                end = data.index [etime]
                data.loc [begin:end, (sensor, ax)] = (
                    np.linspace (data.iloc [begin] [sensor, ax],
                                 data.iloc [end] [sensor, ax],
                                 num = end - begin + 1)
                    )

    if lp != None:
        columns = list ((sensor, axis) for sensor in SENSORS
                        for axis in AXES)
        if hp != None:
            b, a = sig.butter (4, [hp / (fs / 2), lp / (fs / 2)], btype='bandpass')
        else:
            b, a = sig.butter (4, lp / (fs / 2))
        out = sig.filtfilt (b, a, data [columns].to_numpy ().T).astype (int)
        data [columns] = out.T

    return data

from glob import glob
def load_LFP_data(session_folder, foldName = 'binLFPN', loadtype=np.int16):
    LFP_dict = {}
    for file in glob(os.path.join(session_folder, foldName, '*.bin') ):
        elec = file[file.find('Elec')+4:file.find('-F')]
        # print(elec)
        with open(file, 'rb') as f:
            LFP_dict[elec] = np.fromfile(f, dtype=loadtype)
    return LFP_dict


def find_threshold_crossings(time_series:np.array, thresh=-2.8, filter=True, area=120):
    """Find where the time series values cross a threshold

    Args:
        time_series (np.array): time_series, could be LFP data or rate (for instance)
        thresh (float, optional): number of standard deviations from the mean as the threshold. Defaults to -2.8.

    Returns:
        np.array: indices in time series where the threshold was crossed
    """
    thresh = thresh*np.std(time_series) + np.mean(time_series)
    crossings = np.where(np.diff(time_series<thresh )==1)[0]
    if filter:
        crossings = filter_close_crossings(crossings, area=area)
    return crossings


def filter_close_crossings(crossings: np.array, area=150):
    """Remove multiple crossings in the same general area. only keep the first

    Args:
        crossings (np.array): array of positions in the time series where a threshold was crossed. 
        area (int): under this number of indices will be counted as close

    Returns:
        np.array: crossings after being filtered
    """
    i=1
    while i < len(crossings):
        if crossings[i] - crossings[i-1] < area:
            crossings = np.delete(crossings, i)
        else:
            i +=1
    return crossings


def cut_window(crossings:np.array, time_series:np.array, window=(-50,150)):
    """Gather waveforms from time series around indices in crossings

    Args:
        crossings (np.array): indices of threshild crossings in the time series
        time_series (np.array): time series (LFP or rates for instance)
        window (tuple, optional): values for a window aroud the index window[1]>window[0]. Defaults to (-50,150).

    Returns:
        crossings: after checking and removing indices from the edges
        waveforms: np array of size (len(crossings), window[1]-window[0]) containing waveforms from the time series around indices in crossings
    """
    # remove first and/or last crossings if a window around the exceeds the time series:
    crossings = crossings[crossings > -window[0]]
    crossings = crossings[crossings < (len(time_series) - window[1])]
    # create empty array and fill it:
    waveforms = np.zeros((len(crossings), window[1]-window[0]))
    for i, cross in enumerate(crossings):
        waveforms[i,:] = time_series[cross+window[0]:cross+window[1] ]
    
    return crossings, waveforms

def find_crossings_correlation(template, time_series, thresh=5):
    corr_to_spike = np.convolve(time_series, np.flip(template), mode='same')
    crossings = find_threshold_crossings(corr_to_spike, thresh=thresh, filter=True, area=120)
    return crossings



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', filename='preprocess log', level=logging.DEBUG)
    logging.debug('using main of preprocessing')
    # Enter correct inputs here
    dataDir = "C:\\Data\\K6\\2020-03-19a\\WL\\"
    elecList = list(range(2, 33))
    fileList = list(range(0, 64))
    rangeStr = "-F{0}T{1}".format(fileList[0], fileList[-1])
    remScaledMedian(os.path.join(dataDir, 'binBand'), os.path.join(dataDir, 'binMedTest'), elecList, rangeStr,
                    batchSize=1000000, verbose=True)

# "{0}Elec{1}Motion.bin"
# wirelessToMotion('/mnt/hgfs/vmshared/WLnew/','/mnt/hgfs/vmshared/WLnew/bin/',list(range(99,150)))

# wirelessToBin('/mnt/hgfs/vmshared/WLnew/','/mnt/hgfs/vmshared/WLnew/binNew/',list(range(99,102)),[3,9,31])
