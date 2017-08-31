import numpy as np
import json
import sys
import scipy.io
import math

with open('Li-cmpd-data-new.json') as f:
	data = json.load(f)

mps = np.genfromtxt('mpswoNaN.txt', usecols=0, dtype = 'string')
features = np.loadtxt('featureswoNaN.txt')

def generate_Xdata(symmetry):
	xdata = []
	mps_data = []
	inpfile = 'non-training-groups/'+symmetry + '-non-training-set-data.json'
	with open(inpfile, 'r') as f:
		data = json.load(f)
		for index, elem in enumerate(data):
			for index1, (mp, feature) in enumerate(zip(mps,features)):
				if mp == elem['task_id']:
#					xdata.append(feature)
					xdata.append(np.hstack((feature[0:12],feature[13:16],feature[18:])))
	scipy.io.savemat(symmetry+'-non-training-data',mdict={'mps':mps_data,'xntdata':xdata})

#from constants import cubic_equal, hex_equal, monoclinic_equal, ortho_equal, tetra1_equal, tetra2_equal, trig1_equal, trig2_equal, triclinic_equal, asym_mp

generate_Xdata(symmetry = 'cubic')
generate_Xdata(symmetry = 'hexagonal')
generate_Xdata(symmetry = 'orthorhombic')
generate_Xdata(symmetry = 'monoclinic')
generate_Xdata(symmetry = 'tetragonal-1')
generate_Xdata(symmetry = 'tetragonal-2')
generate_Xdata(symmetry = 'trigonal-2')
generate_Xdata(symmetry = 'trigonal-2')
generate_Xdata(symmetry = 'triclinic')