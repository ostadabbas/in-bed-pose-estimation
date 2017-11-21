import scipy.io as sio
import numpy as np
import json
import cv2
import lmdb
import caffe
import os.path
import struct

# from json file generate LMDB  (python database) perhaps matlab no such interface
# meta_data store state annotations.  width and height as the image
# concascade to 4 channel 
# MANNE SC0.2 edition scale to 0.2 
# datasets: multiple datasets.   
def writeLMDB(datasets, lmdb_path, validation):
	env = lmdb.open(lmdb_path, map_size=int(1e12))
	txn = env.begin(write=True) # open dataset 
	data = []

	for d in range(len(datasets)): # the dataset length, put all in data
		if(datasets[d] == "MPI"):
			print datasets[d]
			with open('json/MPI_annotations.json') as data_file:
				data_this = json.load(data_file)
				data_this = data_this['root'] # get from the data
				data = data + data_this # dataset a list, combine them together 
			numSample = len(data)
			print numSample
		elif(datasets[d] == "LEEDS"):
			print datasets[d]
			with open('json/LEEDS_annotations.json') as data_file:
				data_this = json.load(data_file)
				data_this = data_this['root']
				data = data + data_this
			numSample = len(data)
			print numSample
		elif(datasets[d] == "FLIC"):
			datasets[d]
			with open('json/FLIC_annotations.json') as data_file:
				data_this = json.load(data_file) # from json to current data
				data_this = data_this['root'] # but what is the structure
				data = data + data_this
			numSample = len(data)
			print numSample
			# ------------- give dataset json name here. 
		elif(datasets[d] == "MANNE_GRAY"):
			datasets[d]
			# with open('json/MANNE_GRAY_annotations.json') as data_file:
			with open('json/MANNE_GRAYSC0_2_annotations.json') as data_file:
				data_this = json.load(data_file) # from json to current data
				data_this = data_this['root'] # but what is the structure
				data = data + data_this
			numSample = len(data)
			print numSample

		elif(datasets[d] == "IRS_REAL"):
			datasets[d]
			# with open('json/MANNE_GRAY_annotations.json') as data_file:
			with open('json/IRS_REAL_SC0_2_annotations.json') as data_file:
				data_this = json.load(data_file) # from json to current data
				data_this = data_this['root'] # but what is the structure
				data = data + data_this
			numSample = len(data)
			print numSample
	
	random_order = np.random.permutation(numSample).tolist() # random sequence indexes
	
	isValidationArray = [data[i]['isValidation'] for i in range(numSample)];
	if(validation == 1):
		totalWriteCount = isValidationArray.count(0.0);	# how many zeros non validation data
	else:
		totalWriteCount = len(data)
	print 'going to write %d images..' % totalWriteCount;
	writeCount = 0

	for count in range(numSample):	# 1000? 
		idx = random_order[count]
		if (data[idx]['isValidation'] != 0 and validation == 1):
			print '%d/%d skipped' % (count,idx)
			continue
		#print idx

		# ------------------give the image root here
		if "MPI" in data[idx]['dataset']:
			path_header = '../dataset/MPI/images/'
		elif "LEEDS" in data[idx]['dataset']:
			path_header = '../dataset/LEEDS/'
		elif "FLIC" in data[idx]['dataset']:
			path_header = '../dataset/FLIC/'
		elif "MANNE_GRAY" in data[idx]['dataset']:
			# path_header = '../dataset/MANNE_GRAY'
			path_header = '../dataset/MANNE_GRAY_SC0_2'	# to find images 
		elif "IRS_REAL" in data[idx]['dataset']:
		# path_header = '../dataset/MANNE_GRAY'
			path_header = '../dataset/IRS_REAL_MltSC0_2_170518N'	# to find images 



		
		img = cv2.imread(os.path.join(path_header, data[idx]['img_paths'])) # read in file names
		height = img.shape[0]
		width = img.shape[1]	# original size , too large
		if(width < 64):
			img = cv2.copyMakeBorder(img,0,0,0,64-width,cv2.BORDER_CONSTANT,value=(128,128,128))
			print 'saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
			cv2.imwrite('padded_img.jpg', img)
			width = 64 # top bottomm left right 
			# no modify on width, because we want to keep information
		meta_data = np.zeros(shape=(height,width,1), dtype=np.uint8)
		#print type(img), img.shape
		#print type(meta_data), meta_data.shape
		clidx = 0 # current line index
		# dataset name (string)
		for i in range(len(data[idx]['dataset'])):	# idx in random 1xnSamp 
			meta_data[clidx][i] = ord(data[idx]['dataset'][i])  # get dataset name, give unicode
		clidx = clidx + 1	# first line the dataset 
		# image height, image width 
		height_binary = float2bytes(data[idx]['img_height'])
		for i in range(len(height_binary)):
			meta_data[clidx][i] = ord(height_binary[i])	# each image a height 
		width_binary = float2bytes(data[idx]['img_width'])
		for i in range(len(width_binary)):
			meta_data[clidx][4+i] = ord(width_binary[i]) # back to ASCII number why 4 
		clidx = clidx + 1
		# (a) isValidation(uint8), numOtherPeople (uint8), people_index (uint8), annolist_index (float), writeCount(float), totalWriteCount(float)
		meta_data[clidx][0] = data[idx]['isValidation']
		meta_data[clidx][1] = data[idx]['numOtherPeople']
		meta_data[clidx][2] = data[idx]['people_index']
		annolist_index_binary = float2bytes(data[idx]['annolist_index'])	# image ind
		for i in range(len(annolist_index_binary)): # 3,4,5,6
			meta_data[clidx][3+i] = ord(annolist_index_binary[i])
		count_binary = float2bytes(float(writeCount)) # note it's writecount instead of count! from 0
		for i in range(len(count_binary)):
			meta_data[clidx][7+i] = ord(count_binary[i])
		totalWriteCount_binary = float2bytes(float(totalWriteCount))	# how many data
		for i in range(len(totalWriteCount_binary)):
			meta_data[clidx][11+i] = ord(totalWriteCount_binary[i])
		nop = int(data[idx]['numOtherPeople'])
		clidx = clidx + 1
		# (b) objpos_x (float), objpos_y (float)
		objpos_binary = float2bytes(data[idx]['objpos'])
		for i in range(len(objpos_binary)):
			meta_data[clidx][i] = ord(objpos_binary[i])
		clidx = clidx + 1
		# (c) scale_provided (float)
		scale_provided_binary = float2bytes(data[idx]['scale_provided'])
		for i in range(len(scale_provided_binary)):
			meta_data[clidx][i] = ord(scale_provided_binary[i])
		clidx = clidx + 1
		# (d) joint_self (3*16) or (3*22) (float) (3 line)
		joints = np.asarray(data[idx]['joint_self']).T.tolist() # transpose to 3*16
		for i in range(len(joints)):
			row_binary = float2bytes(joints[i])
			for j in range(len(row_binary)):
				meta_data[clidx][j] = ord(row_binary[j])
			clidx = clidx + 1
		# (e) check nop, prepare arrays
		if(nop!=0):
			if(nop==1):
				joint_other = [data[idx]['joint_others']]
				objpos_other = [data[idx]['objpos_other']]
				scale_provided_other = [data[idx]['scale_provided_other']]
			else:
				joint_other = data[idx]['joint_others']
				objpos_other = data[idx]['objpos_other']
				scale_provided_other = data[idx]['scale_provided_other']
			# (f) objpos_other_x (float), objpos_other_y (float) (nop lines)
			for i in range(nop):
				objpos_binary = float2bytes(objpos_other[i])
				for j in range(len(objpos_binary)):
					meta_data[clidx][j] = ord(objpos_binary[j])
				clidx = clidx + 1
			# (g) scale_provided_other (nop floats in 1 line)
			scale_provided_other_binary = float2bytes(scale_provided_other)
			for j in range(len(scale_provided_other_binary)):
				meta_data[clidx][j] = ord(scale_provided_other_binary[j])
			clidx = clidx + 1
			# (h) joint_others (3*16) (float) (nop*3 lines)
			for n in range(nop):
				joints = np.asarray(joint_other[n]).T.tolist() # transpose to 3*16
				for i in range(len(joints)):
					row_binary = float2bytes(joints[i])
					for j in range(len(row_binary)):
						meta_data[clidx][j] = ord(row_binary[j])
					clidx = clidx + 1
		
		# print meta_data[0:12,0:48] 
		# total 7+4*nop lines
		img4ch = np.concatenate((img, meta_data), axis=2) 
		img4ch = np.transpose(img4ch, (2, 0, 1)) # change dim , 4  x h x w 
		#print img4ch.shape
		datum = caffe.io.array_to_datum(img4ch, label=0)
	# 	env = lmdb.open(lmdb_path, map_size=int(1e12))
	# txn = env.begin(write=True) # open dataset 
		key = '%07d' % writeCount
		txn.put(key, datum.SerializeToString())	# each key a writeCount number
		if(writeCount % 1000 == 0):	# every 1000 commit
			txn.commit()
			txn = env.begin(write=True)
		print 'count: %d/ write count: %d/ randomized: %d/ all: %d' % (count,writeCount,idx,totalWriteCount)
		writeCount = writeCount + 1

	txn.commit()
	env.close()

def float2bytes(floats):
	if type(floats) is float:
		floats = [floats]
	return struct.pack('%sf' % len(floats), *floats)

if __name__ == "__main__":
	
	# ----------- main calling function to select dataset and set the lmdb file name 
	# for the mannequin data sc02 
	writeLMDB(['MANNE_GRAY'],'lmdb/MANNE_GRAY_SC0_2',0);
	# for IRS_REAL data 
	# writeLMDB(['IRS_REAL'], 'lmdb/IRS_REAL_SC0_2',0);
