#!/usr/bin/python

import numpy as np
import h5py
import struct, sys, os
import PYlog
import copy
import multiprocessing as mp

def procS(file_name):
	ulog2h5py(file_name)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python deh5py.py <log.hdf5>\n")
		sys.exit()

	if (os.path.isdir(sys.argv[1])):
		datafilenameList = []
		logfilenameList = []
		processes = []
		poo = mp.Pool(processes=3,maxtasksperchild=10)
		# is directory, look for all files inside it
		for root, dirs, files in os.walk(sys.argv[1]):
			for file in files:
				if file.endswith('.ulg'):
					fn = os.path.join(root, file)
					print('Processing file %s' % fn)
					datafilename = os.path.splitext(fn)[0] + '.hdf5'
					logfilename = os.path.splitext(fn)[0] + '.ulg'
					if (os.path.isfile(datafilename)):
						pass
					else:
						datafilenameList.append(datafilename)
						logfilenameList.append(logfilename)
						poo.apply(procS, (logfilename,))

		poo.map(procS, logfilenameList)
		poo.close()
		poo.join()

	elif(os.path.isfile(sys.argv[1])):
		fn = sys.argv[1]
		datafilename = os.path.splitext(fn)[0] + '.hdf5'
		logfilename = os.path.splitext(fn)[0] + '.px4log'
		if (os.path.isfile(datafilename)):
			pass
		else:
			parser = PYlog.sdlog2_pp()
			parser.process(logfilename)

	M = h5py.File(datafilename, 'r')

	for topic in M.values():
		for Id in topic.values():
			for field in Id.values():
				try:
					exec('%s = , field.value' % (field.name.replace('/', '_')))
				except:
					print('Error executing field.name = field.value')
