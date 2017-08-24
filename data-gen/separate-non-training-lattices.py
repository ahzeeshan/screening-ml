import json
import numpy as np 
import pymatgen 
import pdb

with open('non-training-set-data.json') as ts:
    ts_data = json.load(ts)
    
sz = len(ts_data)
cubic_training_set = []
monoclinic_ts = []
orthorhombic_ts = []
hexagonal_ts = []
tetragonal_ts1 = []
tetragonal_ts2 = []
triclinic_ts = []
trigonal_ts1 = []
trigonal_ts2 = []
index = 0

while index<sz:
    st = pymatgen.Structure.from_str(ts_data[index]['cif'], fmt = "cif")
    finder = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(st)
    sys = finder.get_crystal_system() # gives unicode data    
    var = finder.get_point_group_symbol()
    
    
    sys_name = sys.encode('ascii','ignore')
    
    if sys_name == 'cubic':
        cubic_training_set.append(ts_data[index])
          
    elif sys_name == 'monoclinic' :
        monoclinic_ts.append(ts_data[index])
        
    elif sys_name == 'orthorhombic':
        orthorhombic_ts.append(ts_data[index])
            
    elif sys_name == 'triclinic':
        triclinic_ts.append(ts_data[index])
            
    elif sys_name == 'hexagonal':
        hexagonal_ts.append(ts_data[index])
            
    elif sys_name == 'tetragonal':
        if var == '4mm' or var == '-42m' or var == '422' or var == '4/mmm': 
            tetragonal_ts1.append(ts_data[index])
        elif var == '4' or var == '-4' or var == '4/m':
            tetragonal_ts2.append(ts_data[index])
        else:
            print 'Some problem tetragonal' 
            
    elif sys_name == 'trigonal':
        if var == '32' or var == '-3m' or var == '3m':
            trigonal_ts1.append(ts_data[index]) 
        elif var == '3' or var == '-3':
            trigonal_ts2.append(ts_data[index])
        else: 
            print 'Some problem trigonal'
    else:
        print "Oops!!" #Just if any case if the above cases are not covered 
    index = index+1

with open('cubic-non-training-set-data.json','w') as ctd:
    json.dump(cubic_training_set,ctd)
    
with open('monoclinic-non-training-set-data.json','w') as mtd:
    json.dump(monoclinic_ts,mtd)
    
with open('orthorhombic-non-training-set-data.json','w') as  otd:
    json.dump(orthorhombic_ts,otd)
    
with open('hexagonal-non-training-set-data.json','w') as htd:
    json.dump(hexagonal_ts,htd)
    
with open('tetragonal-non-training-set-data-0-5-zero.json','w') as ttd:
    json.dump(tetragonal_ts1,ttd)
with open('tetragonal-non-training-set-data-0-5-non-zero.json','w') as ttd:
    json.dump(tetragonal_ts2,ttd)
    
with open('triclinic-non-training-set-data.json','w') as tctd:
    json.dump(triclinic_ts,tctd)
    
with open('trigonal-non-training-set-data-0-4-zero.json','w') as ttgd:
    json.dump(trigonal_ts1,ttgd)    
with open('trigonal-non-training-set-data-0-4-non-zero.json','w') as ttgd:
    json.dump(trigonal_ts2,ttgd)