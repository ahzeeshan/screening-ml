import json
import math
import numpy as np
import itertools
import pdb

def check(symmetry, subsets):
	with open(symmetry+'-training-set-data.json') as f:
		data = json.load(f)
	print '{} {}'.format(symmetry,len(data))
	filename = symmetry+'-non-zeros.txt'
	with open(filename, 'w') as f:
		f.write('i j val index mp formula\n')
	for index, elem in enumerate(data):
		elastic_mat = np.array(elem['elasticity']['elastic_tensor'])
		assert np.allclose(elastic_mat.transpose() , elastic_mat)
		for subset in itertools.combinations_with_replacement([1,2,3,4,5,6],2):
			if math.fabs(elastic_mat[subset[0]-1][subset[1]-1]) > 1e-5 and subset not in subsets:
				with open(filename,'a') as f:
					f.write('{} {} {} {} {} {}\n'.format(subset[0],subset[1],elastic_mat[subset[0]-1][subset[1]-1],index,elem['task_id'],elem['pretty_formula']))


cubic_subsets = [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(4,4),(5,5),(6,6)]
hex_subsets = [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(4,4),(5,5),(6,6)]
monoclinic_subsets = [(1,1),(1,2),(1,3),(1,5),(2,2),(2,3),(2,5),(3,3),(3,5),(4,4),(4,6),(5,1),(5,2),(5,3),(5,5),(6,4),(6,6)]
ortho_subsets = [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(4,4),(5,5),(6,6)]
tetra1_subsets = [(1,1),(1,2),(1,3),(1,6),(2,2),(2,3),(2,6),(3,3),(4,4),(5,5),(6,6)]
tetra2_subsets = [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(4,4),(5,5),(6,6)]
trig1_subsets = [(1,1),(1,2),(1,3),(1,4),(1,5),(2,2),(2,3),(2,4),(2,5),(3,3),(4,4),(4,6),(5,5),(5,6),(6,6)]
trig2_subsets = [(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,3),(4,4),(4,6),(5,5),(5,6),(6,6)]


check(symmetry = 'cubic', subsets = cubic_subsets)
check(symmetry = 'hexagonal', subsets = hex_subsets)
check(symmetry = 'orthorhombic', subsets = ortho_subsets)
check(symmetry = 'monoclinic', subsets = monoclinic_subsets)
check(symmetry = 'tetragonal-1', subsets = tetra1_subsets)
check(symmetry = 'tetragonal-2', subsets = tetra2_subsets)
check(symmetry = 'trigonal-1', subsets = trig1_subsets)
check(symmetry = 'trigonal-2', subsets = trig2_subsets)