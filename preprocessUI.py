import glob
import os
from tempfile import NamedTemporaryFile
import shutil
import csv
import preprocess as pp
import matplotlib.pyplot


# files location = D:\Users\Matan\Downloads\preprocess files test\
# The function gets a row as a dictionary and adds/updates the DB by it
def updateDB(dictionary):
    # sets all the fields of the Data Base
    fields = ["Animal", "Date", "lfp", "Bandpass", "median", "files recorded",
              "Bad electrodes", "Crosstalk", "possible spiking channels", "Neurons after sorting"]
    # The Data Base's path
    DBPath = "D:\\Users\\Matan\\Downloads\\preprocess files test\\ThemisDB.csv"
    # Creates temporary file
    tempFile = NamedTemporaryFile(mode='w', delete=False, newline='')
    # reads from the DB
    with open(DBPath, 'r')as csvFile:
        writer = csv.DictWriter(tempFile, fieldnames=fields)
        reader = csv.DictReader(csvFile, fieldnames=fields)
        isUpdated = False
        for row in reader:
            # if the row exists in the DB, update it
            if row["Animal"] == str(dictionary.get("Animal")) and row["Date"] == str(dictionary.get("Date")):
                for key in dictionary.keys():
                    row.update({key: dictionary.get(key)})
                isUpdated = True
            # write the row to the temp file
            writer.writerow(row)
        # if the row does not exist in the DB, add it
        if not isUpdated:
            writer.writerow(dictionary)
    # close the temp file and replace it in the DB folder
    tempFile.close()
    shutil.move(tempFile.name, DBPath)
    print("ThemisDB.csv has been updated")


# The function converts wireless data to bin data and updates the DB
def wirelessToDB():
    pp.wirelessToBin(inDir, inDir + "binNew\\", fileList, elecList)
    basicRow.update({"files recorded": fileList.__len__()})
    updateDB(basicRow)
    print(str(fileList.__len__()) + " files has been converted from wireless to bin successfully")


# The function converts wireless bin data to lfp bin data and updates the DB
def lfpToDB():
    fileFormat = '{0}Elec{1}' + rangeStr + '.bin'
    pp.binToLFP(inDir + "binNew\\", inDir + "binLFP\\", fileFormat, elecList)
    basicRow.update({"lfp": elecList.__len__()})
    updateDB(basicRow)
    print(str(elecList.__len__()) + " files has been converted from lfp to bin successfully")


# the function filters wireless bin data to bandpass bin data and updates the DB
def bandpassToDB():
    fileFormat = "Elec{0}" + rangeStr + ".bin"
    pp.bandpass_filter(inDir + "binNew\\", inDir + "binBand\\", fileFormat, elecList)
    basicRow.update({"Bandpass": elecList.__len__()})
    updateDB(basicRow)
    print(str(elecList.__len__()) + " files has been converted from bandpass to bin successfully")


# The function shows a single electrode data by path
def showElectrode(path, elecNumber, plotLimit, title):
    fig, axes = pp.plotBin(path, plotLimit)
    axes.set_title('Electrode ' + str(elecNumber) + title, fontsize=20)
    fig.set_size_inches((30, 5))
    matplotlib.pyplot.show()


# The function filters good/bad electrodes by user input.
# Each electrode is shown to the user and he enters good/bad
# The DB gets updated after
def filterGoodBad():
    badElecList = []
    # Present all electrodes
    for elec in elecList:
        showElectrode(os.path.join(inDir + "binNew", "Elec" + str(elec) + rangeStr + ".bin"), elec, [10.05, 10.2], "")
        # Ask for an answer from the user
        result = input("Enter good/bad: ")
        if str(result) == "bad":
            badElecList.append(elec)
    # remove all bad electrodes from elecList
    for badElec in badElecList:
        elecList.remove(badElec)
    # updates the bad electrodes
    basicRow.update({"Bad electrodes": badElecList})
    updateDB(basicRow)


# The function asks the user to analyze crosstalk of electrodes and asks him to enter them.
# The DB gets updated after
def crossTalkToDB():
    ccr, ccf = pp.plot_corr_mat(inDir, rangeStr, elecList, raw_fold="binNew")
    matplotlib.pyplot.show()
    result = input("Enter channels with crosstalk (a,b,c...) or write stop to finish: ")
    lisOfNumLists = []
    while result != "stop":
        # split the result by ','
        strList = result.split(',')
        try:
            # try to convert each string to an int
            numList = list(map(lambda string: int(string), strList))
            lisOfNumLists.append(numList)
        except ValueError:
            # if not all strings can be converted, print an error message
            print("Not all values are numbers! Please enter again")
        # ask for a user input again
        result = input("Enter channels with crosstalk (a,b,c...) or write stop to finish: ")
    # update the DB
    basicRow.update({"Crosstalk": lisOfNumLists})
    updateDB(basicRow)


# The function removes the median from the good electrodes and updates the DB
def removeMedian():
    pp.remScaledMedian(inDir + "binBand\\", inDir + "binMed\\", elecList, rangeStr)
    basicRow.update({"median": elecList.__len__()})
    updateDB(basicRow)
    print(str(elecList.__len__()) + " files removed their median successfully")


def plot5Sec(filePath, axes, elec, samplingRate=32000):
    maxX = axes[1].dataLim.intervalx[1]
    time = maxX / samplingRate
    start = [0, 5]
    end = [time - 5, time]
    middle = [(time / 2) - 2.5, (time / 2) + 2.5]
    showElectrode(filePath, elec, start, " Start")
    showElectrode(filePath, elec, middle, " Middle")
    showElectrode(filePath, elec, end, " End")


def spikingFiltering():
    unitsList = []
    for elec in elecList:
        filePath = os.path.join(inDir, "binMed\\", 'Elec' + str(elec) + rangeStr + '.bin')
        axes = pp.plotAllBin(filePath)
        plot5Sec(filePath, axes, elec)
        result = input("Is possible unit? (yes/no)")
        if result == "yes":
            unitsList.append(elec)
    basicRow.update({"possible spiking channels": unitsList})
    updateDB(basicRow)


inDir = input("Enter a Wireless Recordings path:")
# inDir = "D:\\Users\\Matan\\Downloads\\preprocess files test\\"
# count the DT2 files
DT2Files = [f for f in glob.glob(inDir + "\\*.DT2")]
# create a list of numbers from 0 to the number of DT2 files
fileList = list(range(0, DT2Files.__len__()))
# create a list of electrodes numbers from 2 to 33
elecList = list(range(2, 33))
# example for a basic row Data
basicRow = {"Animal": "K6", "Date": "20/03/2020"}
rangeStr = "-F{0}T{1}".format(fileList[0], fileList[-1])

# wirelessToDB()
# lfpToDB()
# bandpassToDB()
filterGoodBad()
# crossTalkToDB()
# removeMedian()
# spikingFiltering()
