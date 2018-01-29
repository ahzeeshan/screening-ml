import numpy as np
import json
import sys
import scipy.io
import math
import pymatgen
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder as VCF
import pdb
from numpy import genfromtxt
import time
t = time.time()

def generate_Xdata():
	mps_data = []
	
	inpfile = 'non-training-set-data.json'
	v_m = 1.3*pow(10,-5) # from Monroe and Newman's 2005 paper 
	with open(inpfile, 'r') as f:
		data = json.load(f)
	N = len(data)
	print N
	volrat_data = np.zeros(N)
	coord_data = np.zeros(N)
	Gvrh = np.zeros(N)
	Kvrh = np.zeros(N)

	#Read nt predictions
	Gs = genfromtxt('shear_modulus.csv',delimiter=',',usecols=1)
	Ks = genfromtxt('bulk_modulus.csv',delimiter=',',usecols=1)
	mpsG = genfromtxt('shear_modulus.csv',delimiter=',',usecols=0,dtype='string')
	mpsK = genfromtxt('bulk_modulus.csv',delimiter=',',usecols=0,dtype='string')

	for ii, elem in enumerate(data):

		if(ii%100==0):
			print ii
		struc = pymatgen.Structure.from_str(elem['cif'],fmt="cif")
		Lisites = [i for i in range(len(struc)) if struc[i].specie.symbol=='Li']
		f = VCF(structure = struc, allow_pathological=True)
		sumc = 0
		for j in Lisites:
			sumc += f.get_coordination_number(j)
		coord_no = sumc/len(Lisites)
		rad_ref = [0.59, 0.76, 0.92] # data taken from http://abulafia.mt.ic.ac.uk/shannon/radius.php?Element=Li
		coord_ref = [4., 6., 8.]
		fit = np.polyfit(coord_ref, rad_ref, 1)
		fit_fn = np.poly1d(fit)
		rad_Li = fit_fn(coord_no)
		NAV = 6.022*pow(10,23) # Avagadro_num 
		vol_plus = (4./3.)*(np.pi)*(pow(rad_Li*pow(10,-10),3))*NAV
		vol_ratio = vol_plus/v_m
		mps_data.append(elem['task_id'])
		volrat_data[ii] = vol_ratio
		coord_data[ii] = coord_no

		flagG = 0
		flagK = 0
		#Getting predictions
		for ind, mp in enumerate(mpsG):
			if elem['task_id']==mp:
				Gvrh[ii] = 10.**Gs[ind]
				flagG = 1
		
		for ind, mp in enumerate(mpsK):
			if elem['task_id']==mp:
				Kvrh[ii] = 10.**Ks[ind]
				flagK = 1

		if(flagG==0):
			Gvrh[ii] = np.nan
		if(flagK==0):
			Kvrh[ii] = np.nan

	scipy.io.savemat('non-train-data',mdict={'mps':mps_data,'coord':coord_data,'volrat':volrat_data,'Gvrh':Gvrh,'Kvrh':Kvrh})
	if (len(mps_data)!=len(Gvrh)) or (len(mps_data)!=len(Kvrh)):
		print 'The sizes of one or more of G or K are not equal to that of mp'
generate_Xdata()
elapsed = time.time() - t
print elapsed