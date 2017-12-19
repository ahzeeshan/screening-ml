import numpy as np
import json
import sys
import scipy.io
import math
import pymatgen
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder as VCF
import pdb

with open('Li-cmpd-data-Dec18.json') as f:
	data = json.load(f)

mps = np.genfromtxt('mpswoNaN.txt', usecols=0, dtype = 'string')
features = np.loadtxt('featureswoNaN.txt')

def generate_Xdata(symmetry):
	xdata = []
	mps_data = []
	volrat_data = []
	coord_data = []
	inpfile = 'non-training-groups/'+symmetry + '-non-training-set-data.json'
	v_m = 1.3*pow(10,-5) # from Monroe and Newman's 2005 paper 
	with open(inpfile, 'r') as f:
		data = json.load(f)
		for index, elem in enumerate(data):
			for index1, (mp, feature) in enumerate(zip(mps,features)):
				if mp == elem['task_id']:
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
					xdata.append(np.hstack((feature[0:12],feature[13:16],feature[18:])))
					mps_data.append(mp)
					volrat_data.append(vol_ratio)
					coord_data.append(coord_no)
	scipy.io.savemat(symmetry+'-non-training-data',mdict={'mps':mps_data,'xntdata':xdata,'coord':coord_data,'volrat':volrat_data})

#from constants import cubic_equal, hex_equal, monoclinic_equal, ortho_equal, tetra1_equal, tetra2_equal, trig1_equal, trig2_equal, triclinic_equal, asym_mp

generate_Xdata(symmetry = 'cubic')
generate_Xdata(symmetry = 'hexagonal')
generate_Xdata(symmetry = 'orthorhombic')
generate_Xdata(symmetry = 'monoclinic')
generate_Xdata(symmetry = 'tetragonal-1')
generate_Xdata(symmetry = 'tetragonal-2')
generate_Xdata(symmetry = 'trigonal-1')
generate_Xdata(symmetry = 'trigonal-2')
generate_Xdata(symmetry = 'triclinic')
