import numpy as np
import json
import sys
import scipy.io
import math
import pymatgen
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder as VCF

with open('Li-cmpd-data-new.json') as f:
	data = json.load(f)

mps = np.genfromtxt('mpswoNaN.txt', usecols=0, dtype = 'string')
features = np.loadtxt('featureswoNaN.txt')

def generate_XYdata(symmetry, equals):
	xdata = []
	ydata = []
	volrat_data = []
	coord_data = []
	Gr=[]
	Gv=[]
	Gvrh=[]
	Kr=[]
	Kv=[]
	Kvrh=[]
	mps_data = []
	v_m = 1.3*pow(10,-5)
	inpfile = 'training-groups/'+symmetry + '-training-set-data.json'
	with open(inpfile, 'r') as f:
		data = json.load(f)
		for index, elem in enumerate(data):
			for index1, (mp, feature) in enumerate(zip(mps,features)):
				store_mat = True
				for a in feature:
					if math.isnan(a):
						store_mat = False
				elastic_mat = elem['elasticity']['elastic_tensor']
				if not(np.all(np.linalg.eigvals(elastic_mat) > 0)):
					store_mat = False
				if mp == elem['task_id'] and mp not in asym_mp and store_mat:
					mps_data.append(mp)
					#xdata.append(feature)

					struc = pymatgen.Structure.from_str(data[index]['cif'],fmt="cif")
					Lisites = [i for i in range(len(struc)) if struc[i].specie.symbol=='Li']
					f = VCF(structure = struc, allow_pathological=True)
					sum = 0
					for j in Lisites:
						sum += f.get_coordination_number(j)
					
					coord_no = sum/len(Lisites)
					rad_ref = [0.59, 0.76, 0.92] # data taken from http://abulafia.mt.ic.ac.uk/shannon/radius.php?Element=Li
					coord_ref = [4., 6., 8.]
					fit = np.polyfit(coord_ref, rad_ref, 1)
					fit_fn = np.poly1d(fit)
					rad_Li = fit_fn(coord_no)
					NAV = 6.022*pow(10,23) # Avagadro_num 
					vol_plus = (4./3.)*(np.pi)*(pow(rad_Li*pow(10,-10),3))*NAV
					vol_ratio = vol_plus/v_m
					volrat_data.append(vol_ratio)
					coord_data.append(coord_no)

					Gr.append(elem['elasticity']['G_Reuss'])
					Gv.append(elem['elasticity']['G_Voigt'])
					Gvrh.append(elem['elasticity']['G_VRH'])
					Kr.append(elem['elasticity']['K_Reuss'])
					Kv.append(elem['elasticity']['K_Voigt'])
					Kvrh.append(elem['elasticity']['K_VRH'])
					xdata.append(np.hstack((feature[0:12],feature[13:16],feature[18:])))
					elastic_mat = elem['elasticity']['elastic_tensor']
					y = []
					for equal_subsets in equals:
						val = 0
						for subset in equal_subsets:
							val = val + elastic_mat[subset[0]-1][subset[1]-1]
						y.append(val / len(equal_subsets)) #averaged over all equals
					ydata.append(y)
	scipy.io.savemat(symmetry+'-data-posd',mdict={'mps':mps_data,'xdata':xdata,'ydata':ydata,'Gr':Gr,'Gv':Gv,'Gvrh':Gvrh,'Kr':Kr,'Kv':Kv,'Kvrh':Kvrh,'volratt':volrat_data,'coordt':coord_data})

from constants import cubic_equal, hex_equal, monoclinic_equal, ortho_equal, tetra1_equal, tetra2_equal, trig1_equal, trig2_equal, triclinic_equal, asym_mp

generate_XYdata(symmetry = 'cubic', equals = cubic_equal)
generate_XYdata(symmetry = 'hexagonal', equals = hex_equal)
generate_XYdata(symmetry = 'orthorhombic', equals = ortho_equal)
generate_XYdata(symmetry = 'monoclinic', equals = monoclinic_equal)
generate_XYdata(symmetry = 'tetragonal-1', equals = tetra1_equal)
generate_XYdata(symmetry = 'tetragonal-2', equals = tetra2_equal)
generate_XYdata(symmetry = 'trigonal-1', equals = trig1_equal)
generate_XYdata(symmetry = 'trigonal-2', equals = trig2_equal)
generate_XYdata(symmetry = 'triclinic', equals = triclinic_equal)