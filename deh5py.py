#!/usr/bin/python

import numpy as np
import h5py
import struct, sys, os
import copy
import multiprocessing as mp
from pyulog import ULog

def procS(file_name):
	p = ULog(file_name)
	p.ulog2h5py()

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python deh5py.py <log.hdf5>\n")
		sys.exit()

	if (os.path.isdir(sys.argv[1])):
		datafilenameList = []
		logfilenameList = []
		processes = []
		poo = mp.Pool(processes=6,maxtasksperchild=10)
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
						poo.apply_async(procS, (logfilename,))

		# poo.map(procS, logfilenameList)
		poo.close()
		poo.join()

	elif(os.path.isfile(sys.argv[1])):
		fn = sys.argv[1]
		datafilename = os.path.splitext(fn)[0] + '.hdf5'
		logfilename = os.path.splitext(fn)[0] + '.px4log'
		if (os.path.isfile(datafilename)):
			pass
		else:
			p = ULog(logfilename)
			p.ulog2h5py()

	M = h5py.File(datafilename, 'r')
	for topic in M.values():
		for Id in topic.values():
			for field in Id.values():
				try:
					exec('%s = field.value' % (field.name.replace('/', '_').replace('[', '_').replace(']', '')))
				except Exception as e:
					print(e)
					print('Error executing field.name = field.value')
