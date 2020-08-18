import glob
from tempfile import NamedTemporaryFile
import shutil
import csv
import preprocess as pp


# files location = D:\Users\Matan\Downloads\preprocess files test\
# The function gets a row as a dictionary and adds/updates the DB by it
def updateDB(dict):
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
            if row["Animal"] == str(dict.get("Animal")) and row["Date"] == str(dict.get("Date")):
                for key in dict.keys():
                    row.update({key: dict.get(key)})
                isUpdated = True
            # write the row to the temp file
            writer.writerow(row)
        # if the row does not exist in the DB, add it
        if not isUpdated:
            writer.writerow(dict)
    # close the temp file and replace it in the DB folder
    tempFile.close()
    shutil.move(tempFile.name, DBPath)
    print("ThemisDB.csv has been updated")


def wirelessToDB():
    pp.wirelessToBin(inDir, inDir + "binNew\\", fileList, elecList)
    filesNum = fileList.__len__()
    basicRow.update({"files recorded": filesNum})
    updateDB(basicRow)
    print(str(filesNum) + " files has been transformed from wireless to bin successfully")


def lfpToDB():
    fileFormat = '{0}Elec{1}' + rangeStr + '.bin'
    pp.binToLFP(inDir + "binNew\\", inDir + "binLFP\\", fileFormat, elecList)
    basicRow.update({"lfp": elecList.__len__()})
    updateDB(basicRow)
    print(str(elecList.__len__()) + " files has been transformed from lfp to bin successfully")


def bandpassToDB():
    fileFormat = "Elec{0}" + rangeStr + ".bin"
    pp.bandpass_filter(inDir + "binNew\\", inDir + "binBand\\", fileFormat, elecList)
    basicRow.update({"Bandpass": elecList.__len__()})
    updateDB(basicRow)
    print(str(elecList.__len__()) + " files has been transformed from bandpass to bin successfully")


def filterGoodBad():
    pp.plot_channels(inDir, fileList, elecList, raw_fold='binNew')


# inDir = input("Enter a Wireless Recordings path:")
inDir = "D:\\Users\\Matan\\Downloads\\preprocess files test\\"
# count the DT2 files
DT2Files = [f for f in glob.glob(inDir + "\\*.DT2")]
# create a list of numbers from 0 to the number of DT2 files
fileList = list(range(0, DT2Files.__len__()))
# create a list of electrodes numbers from 2 to 33
elecList = list(range(2, 33))
# example for a basic row Data
basicRow = {"Animal": "K6", "Date": "20/03/2020"}
rangeStr = "-F{0}T{1}".format(fileList[0], fileList[-1])

wirelessToDB()
lfpToDB()
bandpassToDB()
filterGoodBad()
