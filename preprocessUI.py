import glob
import os
from tempfile import NamedTemporaryFile
import shutil
import csv
from datetime import date
import preprocess as pp
import matplotlib.pyplot
from github import Github


# The function asks for a data base path and checks if it exists
def getDBPath():
    validPath = False
    while not validPath:
        path = os.path.join(input("Enter Data base path: "), 'ThemisDB.csv')
        try:
            open(path)
            validPath = True
        except OSError:
            print("Couldn't find a data base! please check directories")
    return path


# The function asks for a wireless recordings directory, and checks there are files exists there
# The function also tries to parse the directory and get animal data from it
def getInDir():
    validPath = False
    while not validPath:
        path = os.path.join(input("Enter a Wireless Recordings path: "))
        try:
            files = [f for f in glob.glob(os.path.join(path, "") + "*.DT2")]
            if files.__len__() == 0:
                raise OSError
            getAnimalData(path)
            validPath = True
        except OSError:
            print("Couldn't find wireless recordings! please check directories")
    return path, files


# the function parses the directories to get the animal data
def getAnimalData(filePath):
    directories = filePath.split(os.path.sep)
    try:
        animal = directories[-3]
        sessionDate = date.fromisoformat(directories[-2])
    except ValueError:
        print("Couldn't get animal data! please check directories")
    basicRow.update(
        {"Animal": animal, "Date": str.format("{0}/{1}/{2}", sessionDate.day, sessionDate.month, sessionDate.year)})


# The function gets a row as a dictionary and adds/updates the DB by it
def updateDB(dictionary):
    # sets all the fields of the Data Base
    fields = ["Animal", "Date", "lfp", "Bandpass", "median", "files recorded",
              "Bad electrodes", "Crosstalk", "possible spiking channels", "Neurons after sorting"]
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
    pp.wirelessToBin(os.path.join(inDir, ""), os.path.join(inDir, "binNew"), fileList, elecList)
    basicRow.update({"files recorded": fileList.__len__()})
    updateDB(basicRow)
    print(str(fileList.__len__()) + " files has been converted from wireless to bin successfully")


# The function converts wireless bin data to lfp bin data and updates the DB
def lfpToDB():
    fileFormat = '{0}Elec{1}' + rangeStr + '.bin'
    pp.binToLFP(os.path.join(inDir, "binNew", ""), os.path.join(inDir, "binNew"), fileFormat, elecList)
    basicRow.update({"lfp": elecList.__len__()})
    updateDB(basicRow)
    print(str(elecList.__len__()) + " files has been converted from lfp to bin successfully")


# the function filters wireless bin data to bandpass bin data and updates the DB
def bandpassToDB():
    fileFormat = "Elec{0}" + rangeStr + ".bin"
    pp.bandpass_filter(os.path.join(inDir, "binNew", ""), os.path.join(inDir, "binBand"), fileFormat, elecList)
    basicRow.update({"Bandpass": elecList.__len__()})
    updateDB(basicRow)
    print(str(elecList.__len__()) + " files has been converted from bandpass to bin successfully")


# The function filters good/bad electrodes by user input.
# Each electrode is shown to the user and he enters good/bad
# The DB gets updated after
def goodBadFiltering():
    badElecList = []
    # asks for a second from the user. must be an integer
    isNumber = False
    while not isNumber:
        second = input("Enter a second you wish to see the electrodes from: ")
        try:
            second = int(second)
            isNumber = True
        except ValueError:
            print("Casting problem! Must be an integer")
    # Present all electrodes
    for elec in elecList:
        pp.showElectrode(os.path.join(inDir, "binNew", "Elec" + str(elec) + rangeStr + ".bin"),
                         elec, [int(second), int(second) + 5], "")
        # Ask for an answer from the user
        result = input("Enter good/bad: ")
        while str(result) != "bad" and str(result) != "good":
            if str(result) == "bad":
                badElecList.append(elec)
            elif str(result) != "good":
                print("Type error, please try again")
                result = input("Enter good/bad: ")
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
    pp.remScaledMedian(os.path.join(inDir, "binBand", ""), os.path.join(inDir, "binMed"), elecList, rangeStr)
    basicRow.update({"median": elecList.__len__()})
    updateDB(basicRow)
    print(str(elecList.__len__()) + " files removed their median successfully")


# The function shows 5 seconds from each electrode's start, middle and ending
# And asks the user if it is a possible unit
def spikingFiltering():
    unitsList = []
    for elec in elecList:
        filePath = os.path.join(inDir, "binMed", 'Elec' + str(elec) + rangeStr + '.bin')
        axes = pp.plotAllBin(filePath, elec)
        pp.plot5Sec(filePath, axes, elec)
        result = input("Is a possible unit? (yes/no)")
        while str(result) != "yes" and str(result) != "no":
            if str(result) == "yes":
                unitsList.append(elec)
            elif str(result) != "no":
                result = input("Is a possible unit? (yes/no)")
    basicRow.update({"possible spiking channels": unitsList})
    updateDB(basicRow)


# the function adds/updates the file in the gitHub repository
def addDBToGit():
    # MatanNoach's access token
    g = Github("980c4ec6b0b3f2b801ae468786c7fc4b89433cd5")
    # gets the repository
    repo = g.get_repo("orelTahary/Preprocess")
    oldFile = repo.get_contents("ThemisDB.csv")
    with open(DBPath, 'rb') as fd:
        # reads the DB content as bytes
        contents = fd.read()
        # tries to update the file in the repository
        try:
            repo.update_file(path="ThemisDB.csv", message="Updating DB", content=contents, sha=oldFile.sha)
            print("File ThemisDB.csv updated successfully on github")
        except FileNotFoundError:
            # if the file does not exist, try to add it to the repository
            print("File does not exist on github. creating a new one")
            try:
                repo.create_file(path="ThemisDB.csv", message="Updating DB", content=contents)
                print("File ThemisDB.csv created successfully on github")
            # in case there was a problem, throw an exception
            except RuntimeError:
                print("There was a problem while creating the file on github")


# asks for data base and directory paths
basicRow = {}
DBPath = getDBPath()
inDir, DT2Files = getInDir()
# create a list of numbers from 0 to the number of DT2 files
fileList = list(range(0, DT2Files.__len__()))
# create a rangeStr string format
rangeStr = "-F{0}T{1}".format(fileList[0], fileList[-1])
# create a list of electrodes numbers from 2 to 33
elecList = list(range(2, 33))
# start the whole process
wirelessToDB()
lfpToDB()
bandpassToDB()
goodBadFiltering()
crossTalkToDB()
removeMedian()
spikingFiltering()
addDBToGit()
