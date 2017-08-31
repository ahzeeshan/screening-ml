import numpy as np
import json
import sys
import scipy.io
import math

with open('Li-cmpd-data-new.json') as f:
	data = json.load(f)

mps = np.genfromtxt('mpswoNaN.txt', usecols=0, dtype = 'string')
features = np.loadtxt('featureswoNaN.txt')

def generate_XYdata(symmetry, equals):
	xdata = []
	ydata = []
	mps_data = []
	inpfile = 'training-groups/'+symmetry + '-training-set-data.json'
	with open(inpfile, 'r') as f:
		data = json.load(f)
		for index, elem in enumerate(data):
			for index1, (mp, feature) in enumerate(zip(mps,features)):
				store_mat = True
				for a in feature:
					if math.isnan(a):
						store_mat = False
				if mp == elem['task_id'] and mp not in asym_mp and store_mat:
					mps_data.append(mp)
					#xdata.append(feature)
					xdata.append(np.hstack((feature[0:12],feature[13:16],feature[18:])))
					elastic_mat = elem['elasticity']['elastic_tensor']
					y = []
					for equal_subsets in equals:
						val = 0
						for subset in equal_subsets:
							val = val + elastic_mat[subset[0]-1][subset[1]-1]
						y.append(val / len(equal_subsets)) #averaged over all equals
					ydata.append(y)
	scipy.io.savemat(symmetry+'-data',mdict={'mps':mps_data, 'xdata':xdata, 'ydata':ydata})

from constants import cubic_equal, hex_equal, monoclinic_equal, ortho_equal, tetra1_equal, tetra2_equal, trig1_equal, trig2_equal, triclinic_equal, asym_mp

generate_XYdata(symmetry = 'cubic', equals = cubic_equal)
generate_XYdata(symmetry = 'hexagonal', equals = hex_equal)
generate_XYdata(symmetry = 'orthorhombic', equals = ortho_equal)
generate_XYdata(symmetry = 'monoclinic', equals = monoclinic_equal)
generate_XYdata(symmetry = 'tetragonal-1', equals = tetra1_equal)
generate_XYdata(symmetry = 'tetragonal-2', equals = tetra2_equal)
generate_XYdata(symmetry = 'trigonal-2', equals = trig1_equal)
generate_XYdata(symmetry = 'trigonal-2', equals = trig2_equal)
generate_XYdata(symmetry = 'triclinic', equals = triclinic_equal)