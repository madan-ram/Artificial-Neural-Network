import os, sys
from os import listdir
from os.path import isfile, join

def getFiles(mypath):
	onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]	
	return onlyfiles

count = int(sys.argv[1])
for f in getFiles(sys.argv[2]):
	os.rename(sys.argv[2]+'/'+f, sys.argv[2]+'/'+str(count)+'.jpg')
	count += 1