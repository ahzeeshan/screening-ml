import numpy as np
import json
import sys
import scipy.io
import math
import pymatgen
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder as VCF

def generate_XYtraindata():
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
	inpfile = 'training-set-data.json'
	with open(inpfile, 'r') as f:
		data = json.load(f)

	for elem in data:
		mps_data.append(elem['task_id'])
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
		volrat_data.append(vol_ratio)
		coord_data.append(coord_no)

		Gr.append(elem['elasticity']['G_Reuss'])
		Gv.append(elem['elasticity']['G_Voigt'])
		Gvrh.append(elem['elasticity']['G_VRH'])
		Kr.append(elem['elasticity']['K_Reuss'])
		Kv.append(elem['elasticity']['K_Voigt'])
		Kvrh.append(elem['elasticity']['K_VRH'])
	scipy.io.savemat('train-data-posd',mdict={'mps':mps_data,'Gr':Gr,'Gv':Gv,'Gvrh':Gvrh,'Kr':Kr,'Kv':Kv,'Kvrh':Kvrh,'volratt':volrat_data,'coordt':coord_data})


generate_XYtraindata()