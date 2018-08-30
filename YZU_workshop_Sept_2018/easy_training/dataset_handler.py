'''
    This file is part of the September 2018 Workshop at Yuan Ze University.

    You can use these examples in the way you seem fit, though I can't make sure
    it will work fine in your case.
'''

from os import listdir
from os.path import isfile, join, isdir
import numpy as np

MAX_Tr_N_SAMPLES = 2500 # 250 per class
MAX_Tst_N_SAMPLES = 1000 # 100 per class

def image_paths_retriever(datapath=None, one_hot=True):
	if datapath == None:
		print("No input information given, exiting ...")

	# ===========================================================================
	# training samples first
	training_path = datapath + '/training'

	# get class folders
	classFolders = [f for f in listdir(training_path) if isdir(join(training_path, f))]

	print("%d classes found in %s !!" % (len(classFolders), training_path))

	hot_encoder_train = []
	training_set = []
	
	hot_encoder_test = []
	testing_set = []

	for class_id in range(len(classFolders)):
		# per class
		testing_size = 0
		training_size = 0

		# get class folders
		path2scan = training_path + '/' + classFolders[class_id]
		filesFound = [f for f in listdir(path2scan) if isfile(join(path2scan, f))]
		print("%d files found in %s !!" % (len(filesFound), path2scan))

		for fileid in range(len(filesFound)):
			if training_size < MAX_Tr_N_SAMPLES:
				# training samples
				full_filepath = path2scan + '/' + filesFound[fileid]
				label = np.zeros(shape=[10], dtype=float)
				label[class_id] = 1
				hot_encoder_train.append(label)
				training_set.append(full_filepath)
				training_size += 1
			elif testing_size < MAX_Tst_N_SAMPLES:
				# testing samples
				full_filepath = path2scan + '/' + filesFound[fileid]
				label = np.zeros(shape=[10], dtype=float)
				label[class_id] = 1
				hot_encoder_test.append(label)
				testing_set.append(full_filepath)
				testing_size += 1
			else:
				break

	return training_set, hot_encoder_train, testing_set, hot_encoder_test