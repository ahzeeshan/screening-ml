import numpy as np
import json
import sys
import scipy.io
import math
import pdb

with open('Li-cmpd-data-new.json') as f:
	data = json.load(f)

mps = np.genfromtxt('full_parms_1A_for_EDC.txt', usecols=0, dtype = 'string')
features = np.loadtxt('full_parms_1A_for_EDC.txt', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) )

print features.shape

nan_arr = []
for index, (mp, feature) in enumerate(zip(mps,features)):
	for a in feature:
		if math.isnan(a):
			nan_arr.append(index)
			break

nan_arr = sorted(set(nan_arr), reverse = True)
mps = np.delete(arr = mps,obj = nan_arr,axis = 0)
features = np.delete(arr = features,obj = nan_arr,axis = 0)



#pdb.set_trace()

new_feature = []
mps_new = []
features_new = []
for i, d in enumerate(data):
	for index, (mp, feature) in enumerate(zip(mps, features)):
		if mp == d['task_id']:
			mps_new.append(mp)
			features_new.append(feature)
			Unit_cell = d['unit_cell_formula']
    	    Li_num[index] = Unit_cell['Li']
			new_feature.append(d['formation_energy_per_atom'])
			break

features_new = np.array(features_new)
new_feature = np.transpose(np.array([new_feature]))
print features_new.shape
print new_feature.shape
features_new = np.hstack((features_new, new_feature))
pdb.set_trace()



np.savetxt('mpswoNaN.txt',mps_new, fmt = "%s")
scipy.io.savemat('Xdata-woNaN', mdict={'features':features})
np.savetxt('featureswoNaN.txt',features_new, fmt='%1.6f',delimiter='\t')

'''
1 Volume per atoma 0.20 4.582 13.342 0
2 Standard deviation in Li neighbour count 0.22 1.430 1.766 0
3 Standard deviation in Li bond ionicity 0.04 0.274 0.858 0
4 Li bond ionicitya 0.18 0.372 1.403 0
5 Li neighbour counta 0.19 6.393 21.359 0
6 Li–Li bonds per Lia 0.06 4.432 6.218 +0.817
7 Bond ionicity of sublatticea 0.28 0.330 0.978 1.323
8 Sublattice neighbour counta 0.13 7.087 20.660 0
9 Anion framework coordinationa 0.06 2.202 10.073 1.028
10 Minimum anion–anion separation distancea (Å) 0.09 0.708 3.395 0
11 Volume per anion (Å3) 0.01 35.131 36.614 0
12 Minimum Li–anion separation distancea (Å) 0.20 0.288 2.072 +2.509
13 Minimum Li–Li separation distancea (Å) 0.10 0.746 2.730 1.619
14 Electronegativity of sublatticea 0.16 0.306 2.780 0
15 Packing fraction of full crystal 0.16 0.173 0.465 0
16 Packing fraction of sublattice 0.19 0.186 0.234 0
17 Straight-line path widtha (Å) 0.07 0.247 0.852 0
18 Straight-line path electronegativitya 0.29 0.707 2.535 0
19 Ratio of features (4) and (7) 0.03 0.719 1.611 0
20 Ratio of features (5) and (8) 0.18 0.152 1.057 0
21 formation_energy_per_atom
'''