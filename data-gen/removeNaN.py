import numpy as np
import json
import sys
import scipy.io
import math

with open('Li-cmpd-data-new.json') as f:
	data = json.load(f)
mps = np.genfromtxt('full_parms_1A_for_EDC.txt', usecols=0, dtype = 'string')
features = np.loadtxt('full_parms_1A_for_EDC.txt', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) )

nan_arr = []
for index, (mp, feature) in enumerate(zip(mps,features)):
	for a in feature:
		if math.isnan(a):
			nan_arr.append(index)

nan_arr = sorted(set(nan_arr))
mps = np.delete(arr = mps,obj = nan_arr,axis = 0)
features = np.delete(arr = features,obj = nan_arr,axis = 0)

np.savetxt('mpswoNaN.txt',mps, fmt = "%s")
scipy.io.savemat('Xdata-woNaN', mdict={'features':features})
np.savetxt('featureswoNaN.txt',features)
