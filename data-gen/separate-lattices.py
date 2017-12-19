import json
import numpy as np 
import pymatgen 
import pdb
import os

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

path = 'training-groups/'
nfname = 'negative-eig.txt'
with open(nfname,'w') as f:
            f.write('id\n')

for index, elem_data in enumerate(ts_data):
    st = pymatgen.Structure.from_str(elem_data['cif'], fmt = "cif")
    elastic_mat = elem_data['elasticity']['elastic_tensor']
    analyzer = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(st)
    crys_sys = analyzer.get_crystal_system()
    var = analyzer.get_point_group_symbol()
    if (np.all(np.linalg.eigvals(elastic_mat) > 0)):
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