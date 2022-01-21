# -*- coding: utf-8 -*-
import glob

# -----------------------------------------------------------------------------
def readFilesFromPath(strPath, fileExt = ".wav"):
    listofFiles = []
    for file in glob.glob(strPath + "/" + "*" +fileExt):
        listofFiles.append(file)
    listofFiles.sort()
    return listofFiles