# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import time
import pickle
import os

import numpy as np
print(f"numpy version {np.__version__}")
import matplotlib
import matplotlib.pyplot as plt
print(f"matplotlib version {matplotlib.__version__}")
import pandas as pd
print(f"pandas version {pd.__version__}")
import preprocess as pp
print(f"preprocess version {pp.__version__}")
import scipy.signal as sig
from scipy import __version__ as ver
print(f"scipy version {ver}")

# Enable reloading of packages upon changes
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Enable resizing of Jupyter notebook based on the browser width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# %%
# setting up a logger for the module, 
# by defaut, it would print all warning messages to stdout 
# while also recording debug
import logging

formatter = logging.Formatter('%(asctime)s %(funcName)5s %(levelname)s: %(message)s')
logFile = 'preprocess log'
# logging.basicConfig(format=form)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('preprocessing.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setLevel(logging.WARNING)
sh.setFormatter(formatter)
logger.addHandler(sh)


# %%
# Enter correct inputs here
dataDir = "C:\\Data\\K6\\2020-03-23\\WL\\"
elecList = list(range(2,33))
fileList = list(range(0,231))
rangeStr = "-F{0}T{1}".format(fileList[0], fileList[-1])


# %%
# Convert to bin:
timeBegin = time.time()
lData = pp.wirelessToBin(dataDir,dataDir + 'binNew/',fileList, elecList, verbose=False)
timeEnd = time.time()
print(f"Converted {len(elecList)} electrodes {lData:,d} samples in {timeEnd-timeBegin:.2f} seconds")


# %%
# save LFP downsampled and filtered:
timeBegin = time.time()
pp.binToLFP(dataDir+'binNew/', dataDir+'binLFPN/', '{0}Elec{1}' + rangeStr + '.bin', elecList
                    , freq=[2, 300], notch=True, verbose=False)
timeEnd = time.time()
print(f"Downsampled {len(elecList)} electrodes in {timeEnd-timeBegin:.2f} seconds")


# %%
timeBegin = time.time()
pp.bandpass_filter(os.path.join(dataDir,'binNew'), os.path.join(dataDir,'binBand'), 'Elec{0}' + rangeStr + '.bin', elecList, freq=[300, 6000])
timeEnd = time.time()
print(f"Filtered {len(elecList)} electrodes in {timeEnd-timeBegin:.2f} seconds")


# %%
channels_to_plot = list(range(2,33))
ccr, ccf = pp.plot_corr_mat(dataDir, rangeStr, channels_to_plot, raw_fold='binNew', filt_fold='None', draw_lfp=False)


# %%
# pp.plot_channels(dataDir, fileList, elecList, num_seconds_to_plot=5) #bs -index of first values sampled


# %%
d = (ccf>0.8) & (1-np.eye(len(channels_to_plot))).astype(bool)
e=d.nonzero()
pairs = []
for i in range(0, len(e[0])):
    if (e[0][i]<e[1][i]):
        pairs.append([channels_to_plot[e[0][i]], channels_to_plot[e[1][i]]])
print(f'Problematic electrodes {pairs}')


# %%
elecList = list1 = [4,5] + list(range(8,13)) + list(range(14,17))
# elecList = [2,3,8,9,20,21,22,23,24,25]
timeBegin = time.time()
pp.remMedian(os.path.join(dataDir, 'binNew'), os.path.join(dataDir, 'binMedRaw') , elecList, rangeStr, batchSize=1000000, verbose=False)
timeEnd = time.time()
print(f"Remove median for {len(elecList)} in {timeEnd-timeBegin} seconds ")


# %%
elecList = list2 = list(range(18,26)) + list(range(28,33))
timeBegin = time.time()
pp.remMedian(os.path.join(dataDir, 'binNew'), os.path.join(dataDir, 'binMedRaw') , elecList, rangeStr, batchSize=1000000, verbose=False)
timeEnd = time.time()
print(f"Remove median for {len(elecList)} in {timeEnd-timeBegin} seconds ")


# %%
channels_to_plot = list1 + list2
ccr, ccf = pp.plot_corr_mat(dataDir, rangeStr, channels_to_plot, raw_fold='binMedRaw', filt_fold='None', draw_lfp=False)


# %%
# pp.plot_channels(dataDir, fileList, channels_to_plot, num_seconds_to_plot=5, samplingRate=32000, raw_fold='binMedRaw')


# %%
# elecList = list(range(2,33))
# elecList = [2,3,8,9,20,21,22,23,24,25]
elecList = channels_to_plot #list1 + list2
timeBegin = time.time()
pp.bandpass_filter(os.path.join(dataDir,'binMedRaw'), os.path.join(dataDir,'binMed'), 'Elec{0}' + rangeStr + '.bin', elecList, freq=[300, 6000])
timeEnd = time.time()
print(f"Filtered {len(elecList)} electrodes in {timeEnd-timeBegin:.2f} seconds")


# %%
fig, ax = plt.subplots(figsize=(20,len(elecList)*6), nrows=len(elecList)*2, ncols=1, sharex=True,sharey=True)
plt.ylim([-600,600])
num_seconds_to_plot = 15
samplingRate = 32000
bs = 300000
be = int(bs+samplingRate*num_seconds_to_plot)
t = np.arange(bs, be, 1) / samplingRate
for i, elc in enumerate(elecList):
    rangeStr = "-F{0}T{1}".format(fileList[0], fileList[-1])
    fpath = os.path.join(dataDir, 'binBand', 'Elec' + str(elc) + rangeStr + '.bin')
    elec_data = np.fromfile(fpath,dtype=np.int16)
    ax[i*2].plot(t, elec_data[bs:be])
    ax[i*2].set_title(f'filtered data channel {elc}')
    fpath = os.path.join(dataDir, 'binMed', 'Elec' + str(elc) + rangeStr + '.bin')
    elec_data = np.fromfile(fpath,dtype=np.int16)
    ax[i*2+1].plot(t, elec_data[bs:be])
    ax[i*2+1].set_title(f'filtered+median removed data channel {elc}')


# %%



