import glob
from tempfile import NamedTemporaryFile
import shutil
import csv
import preprocess


# files location = D:\Users\Matan\Downloads\preprocess files test\
# The function gets a row as a dictionary and adds/updates the DB by it
def updateDB(dict):
    # sets all the fields of the Data Base
    fields = ["Animal", "Date", "lfp", "Bandpass", "median", "files recorded",
              "Bad electrodes", "Crosstalk", "possible spiking channels", "Neurons after sorting"]
    # The Data Base's path
    DBPath = "D:\\Users\\Matan\\Downloads\\preprocess files test\\ThemisDB.csv"
    # Creates temporary file
    tempFile = NamedTemporaryFile(mode='w', delete=False)
    # reads from the DB
    with open(DBPath, 'r')as csvFile:
        writer = csv.DictWriter(tempFile, fieldnames=fields)
        reader = csv.DictReader(csvFile, fieldnames=fields)
        isUpdated = False
        for row in reader:
            # if the row exists in the DB, update it
            if row["Animal"] == str(dict.get("Animal")) and row["Date"] == str(dict.get("Date")):
                row = dict
                isUpdated = True
            # write the row to the temp file
            writer.writerow(row)
        # if the row does not exist in the DB, add it
        if not isUpdated:
            writer.writerow(dict)
    # close the temp file and replace it in the DB folder
    tempFile.close()
    shutil.move(tempFile.name, DBPath)


def WLToCsv():
    inDir = input("Enter a Wireless Recordings path:")
    outDir = inDir + "\\binNew"
    # count the DT2 files
    DT2Files = [f for f in glob.glob(inDir + "\\*.DT2")]
    # create a list of numbers from 0 to the number of DT2 files
    fileList = list(range(0, DT2Files.__len__()))
    # create a list of electrodes numbers from 2 to 33
    elecList = list(range(2, 33))
    num = preprocess.wirelessToBin(inDir, outDir, fileList, elecList)
    # example for a row Data
    animal = "K6"
    date = "18/03/2020"
    dictionaryUpdate = {"Animal": animal, "Date": date, "files recorded": fileList.__len__()}
    updateDB(dictionaryUpdate)


WLToCsv()
