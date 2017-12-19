import numpy as np
import json
import sys
import scipy.io
import math
import pymatgen

symmetry='cubic'
inpfile = 'non-training-groups/'+symmetry + '-non-training-set-data.json'

with open(inpfile, 'r') as f:
		data_sym = json.load(f)

#data = scipy.io.loadmat('cubic-data-posd.mat')
data = scipy.io.loadmat('cubic-non-training-data.mat')
fccnos=[]
bccnos=[]
scnos=[]
nonos=[]
for j, mp in enumerate(data['mps']):
	mp = mp.replace(" ","")
	print mp
	for i, elem_data in enumerate(data_sym):
		if(mp==elem_data['task_id']):
			st = pymatgen.Structure.from_str(elem_data['cif'], fmt = "cif")
			analyzer = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(st)
			crys_sys = analyzer.get_crystal_system()
			var = analyzer.get_space_group_symbol()
			print var
			if(var[0]=='F'):
				fccnos.append(j)
			elif(var[0]=='I'):
				bccnos.append(j)
			elif(var[0]=='P'):
				scnos.append(j)
			else:
				nonos.append(j)
			#print


print fccnos
print bccnos
print scnos




