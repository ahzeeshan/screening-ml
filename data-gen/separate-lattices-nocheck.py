import json
import numpy as np 
import pymatgen 
import pdb
import os
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder as VCF
from pymatgen.analysis.elasticity.tensors import Tensor

with open('training-set-data.json') as ts:
    ts_data = json.load(ts)
    
print len(ts_data)
cubic_training_set = []
monoclinic_ts = []
#monoclinic_ts2 = []
orthorhombic_ts = []
hexagonal_ts = []
tetragonal_ts1 = []
tetragonal_ts2 = []
triclinic_ts = []
trigonal_ts1 = []
trigonal_ts2 = []
all_elem_data = []

path = 'training-groups-check/'
nfname = 'negative-eig-chec.txt'
with open(nfname,'w') as f:
            f.write('id\n')
v_m = 1.3*pow(10,-5)

for index, elem_data in enumerate(ts_data):
    st = pymatgen.Structure.from_str(elem_data['cif'], fmt = "cif")
    analyzer = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(st,symprec=0.1)
    elastic_mat = elem_data['elasticity']['elastic_tensor_original']
    elastic_tensor = Tensor.from_voigt(elastic_mat)
    #pdb.set_trace()
    stconv  = analyzer.get_conventional_standard_structure()
    elastic_mat = (elastic_tensor.fit_to_structure(structure=stconv)).voigt
    elastic_mat_sym = (((Tensor.from_voigt(elastic_mat)).voigt_symmetrized).voigt)

    crys_sys = analyzer.get_crystal_system()
    var = analyzer.get_point_group_symbol()
    #getting molar vol ratio


    if (np.all(np.linalg.eigvals(elastic_mat_sym) > 0)):

        Lisites = [i for i in range(len(st)) if st[i].specie.symbol=='Li']
        f = VCF(structure = st, allow_pathological=True)
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
    #volrat_data.append(vol_ratio)
    #coord_data.append(coord_no)
        elem_data['volrat'] = vol_ratio
        elem_data['coord'] = coord_no
        stconv  = analyzer.get_conventional_standard_structure()
        elem_data['conventional_lattice'] = np.ndarray.tolist(stconv.lattice.matrix)
        elem_data['lengths_and_angles'] = stconv.lattice.lengths_and_angles
        elem_data['elasticity']['elastic_tensor_orig_symmetrized'] =       elastic_mat_sym.tolist()
        #pdb.set_trace()
        all_elem_data.append(elem_data)
        if crys_sys == 'cubic':
            cubic_training_set.append(elem_data)
        elif crys_sys == 'monoclinic':
            monoclinic_ts.append(elem_data)
            '''
            if elastic_mat[0][4] != 0:
                monoclinic_ts1.append(elem_data)
            elif elastic_mat[0][4] == 0:
                monoclinic_ts2.append(elem_data)
            else : 
                print 'Some problem monoclinic'
                '''
        elif crys_sys == 'orthorhombic':
            orthorhombic_ts.append(elem_data)
        elif crys_sys == 'triclinic':
            triclinic_ts.append(elem_data)
        elif crys_sys == 'hexagonal':
            hexagonal_ts.append(elem_data)
        elif crys_sys == 'tetragonal':
            if var == '4' or var == '-4' or var == '4/m':
                tetragonal_ts1.append(elem_data)
            elif var == '4mm' or var == '-42m' or var == '422' or var == '4/mmm': 
                tetragonal_ts2.append(elem_data)
            else:
                raise ValueError('No point group found for this tetragonal material {}'.format(elem_data['task_id']))
        elif crys_sys == 'trigonal':
            if var == '3' or var == '-3':
                trigonal_ts1.append(ts_data[index]) 
            elif var == '32' or var == '-3m' or var == '3m':
                trigonal_ts2.append(elem_data)
            else: 
                raise ValueError('No point group found for this trigonal material {}'.format(elem_data['task_id']))
                
        else:
            raise ValueError('No crystal class found for this material {}'.format(elem_data['task_id'])) #Just if any case if the above cases are not covered 

    else:
        print 'Negative eigen value found'
        with open(nfname,'a') as f:
            f.write(elem_data['task_id']+'\n')

with open(path+'cubic-training-set-data.json','w') as ctd:
    json.dump(cubic_training_set,ctd)
    
with open(path+'monoclinic-training-set-data.json','w') as mtd:
    json.dump(monoclinic_ts,mtd)
    
with open(path+'orthorhombic-training-set-data.json','w') as  otd:
    json.dump(orthorhombic_ts,otd)

with open(path+'hexagonal-training-set-data.json','w') as htd:
    json.dump(hexagonal_ts,htd)
    
with open(path+'tetragonal-1-training-set-data.json','w') as ttd:
    json.dump(tetragonal_ts1,ttd)
with open(path+'tetragonal-2-training-set-data.json','w') as ttd:
    json.dump(tetragonal_ts2,ttd)

with open(path+'triclinic-training-set-data.json','w') as tctd:
    json.dump(triclinic_ts,tctd)

with open(path+'trigonal-1-training-set-data.json','w') as ttgd:
    json.dump(trigonal_ts1,ttgd)    
with open(path+'trigonal-2-training-set-data.json','w') as ttgd:
    json.dump(trigonal_ts2,ttgd)
'''with open(path+'all-data.json','w') as ad:
    json.dump(all_elem_data,ad)'''