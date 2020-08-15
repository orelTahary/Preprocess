import glob
import preprocess
import os


def WLToCsv():
    inDir = input("Enter a Wireless Recordings path:")
    outDir = inDir + "\\WirelessToBin"
    dataPaths = [f for f in glob.glob(inDir + "\\*.txt")]
    elecList = []
    # get only the name without the file type
    names = list(map(lambda path: os.path.basename(path).split(".txt")[0], dataPaths))
    preprocess.wirelessToBin(inDir, outDir, names, elecList)


WLToCsv()
