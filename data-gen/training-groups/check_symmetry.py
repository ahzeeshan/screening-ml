import json
import math
import numpy as np
import itertools
import pdb

def check_sym(symmetry, equals, crosses, parities):
	with open(symmetry+'-training-set-data.json') as f:
		data = json.load(f)
	print '{} {}'.format(symmetry,len(data))
	filename = symmetry+'-symmetry-checks.txt'
	with open(filename, 'w') as f:
		f.write('Equalities\n')
		f.write('subsets mp index formula disparity\n')
	#Equalities
	for index, elem in enumerate(data):
		elastic_mat = np.array(elem['elasticity']['elastic_tensor'])
		assert np.allclose(elastic_mat.transpose() , elastic_mat)
		for equal_subsets in equals:
			val = elastic_mat[equal_subsets[0][0]-1][equal_subsets[0][1]-1]
			for subset in equal_subsets:
				if math.fabs( elastic_mat[subset[0]-1][subset[1]-1] - val) > 1e-5:
					with open(filename,'a') as f:
						f.write('{} {} {} {} {}\n'.format(equal_subsets,elem['task_id'],index,elem['pretty_formula'],math.fabs( elastic_mat[subset[0]-1][subset[1]-1] - val)))
	#Crosses
	with open(filename, 'a') as f:
		f.write('\nCrosses\n')
		f.write('(i,j) mp formula disparity\n')
	for index, elem in enumerate(data):
		elastic_mat = np.array(elem['elasticity']['elastic_tensor'])
		val = (elastic_mat[0][0]-elastic_mat[0][1])/2.
		for subset in crosses:
			if math.fabs( elastic_mat[subset[0]-1][subset[1]-1] - val) > 1e-5:
				with open(filename, 'a') as f:
					f.write('{} {} {} {}\n'.format(subset, elem['task_id'], elem['pretty_formula'], elastic_mat[subset[0]-1][subset[1]-1] - val ))

	#Parities
	with open(filename, 'a') as f:
		f.write('\nParities\n')
		f.write('subsets mp formula disparity\n')
	for index, elem in enumerate(data):
		elastic_mat = np.array(elem['elasticity']['elastic_tensor'])
		for parity_subset in parities:
			if math.fabs( elastic_mat[parity_subset[0][0]-1][parity_subset[0][1]-1] + elastic_mat[parity_subset[1][0]-1][parity_subset[1][1]-1] ) > 1e-5:
				with open(filename,'a') as f:
					f.write('{} {} {} {}\n'.format(parity_subset,elem['task_id'],elem['pretty_formula'],math.fabs( elastic_mat[parity_subset[0][0]-1][parity_subset[0][1]-1] + elastic_mat[parity_subset[1][0]-1][parity_subset[1][1]-1] )))


		'''
		for subset in itertools.combinations_with_replacement([1,2,3,4,5,6],2):
			if math.fabs(elastic_mat[subset[0]-1][subset[1]-1]) > 1e-5 and subset not in subsets:
				with open(filename,'a') as f:
					f.write('{} {} {} {} {} {}\n'.format(subset[0],subset[1],elastic_mat[subset[0]-1][subset[1]-1],index,elem['task_id'],elem['pretty_formula']))
'''
cubic_subsets = [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(4,4),(5,5),(6,6)]
hex_subsets = [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(4,4),(5,5),(6,6)]
monoclinic_subsets = [(1,1),(1,2),(1,3),(1,5),(2,2),(2,3),(2,5),(3,3),(3,5),(4,4),(4,6),(5,1),(5,2),(5,3),(5,5),(6,4),(6,6)]
ortho_subsets = [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(4,4),(5,5),(6,6)]
tetra1_subsets = [(1,1),(1,2),(1,3),(1,6),(2,2),(2,3),(2,6),(3,3),(4,4),(5,5),(6,6)]
tetra2_subsets = [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(4,4),(5,5),(6,6)]
trig1_subsets = [(1,1),(1,2),(1,3),(1,4),(1,5),(2,2),(2,3),(2,4),(2,5),(3,3),(4,4),(4,6),(5,5),(5,6),(6,6)]
trig2_subsets = [(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,3),(4,4),(4,6),(5,5),(5,6),(6,6)]

cubic_equal = [[(1,1),(2,2),(3,3)],[(1,2),(1,3),(2,3)],[(4,4),(5,5),(6,6)]]
cubic_cross = []
cubic_parity = []

hex_equal = [[(1,1),(2,2)],[(1,3),(2,3)],[(4,4),(5,5)]]
hex_cross = [(6,6)]
hex_parity = []

monoclinic_equal = []
monoclinic_cross = []
monoclinic_parity = []

ortho_equal = []
ortho_cross = []
ortho_parity = []

tetra1_equal = [[(1,1),(2,2)],[(1,3),(2,3)],[(4,4),(5,5)]]
tetra1_cross = []
tetra1_parity = [[(1,6),(2,6)]]

tetra2_equal = [[(1,1),(2,2)],[(1,3),(2,3)],[(4,4),(5,5)]]
tetra2_cross = []
tetra2_parity = []

trig1_equal = [[(1,1),(2,2)],[(1,3),(2,3)],[(4,4),(5,5)],[(1,4),(5,6)],[(2,5),(4,6)]]
trig1_cross = [(6,6)]
trig1_parity = [[(1,4),(2,4)],[(1,5),(2,5)]]

trig2_equal = [[(1,1),(2,2)],[(1,3),(2,3)],[(4,4),(5,5)],[(1,4),(5,6)]]
trig2_cross = [(6,6)]
trig2_parity = [[(1,4),(2,4)]]

check_sym(symmetry = 'cubic', equals = cubic_equal, crosses = cubic_cross, parities = cubic_parity)
check_sym(symmetry = 'hexagonal', equals = hex_equal, crosses = hex_cross, parities = hex_parity)
check_sym(symmetry = 'orthorhombic', equals = ortho_equal, crosses = ortho_cross, parities = ortho_parity)
check_sym(symmetry = 'monoclinic', equals = monoclinic_equal, crosses = monoclinic_cross, parities = monoclinic_parity)
check_sym(symmetry = 'tetragonal-1', equals = tetra1_equal, crosses = tetra1_cross, parities = tetra1_parity)
check_sym(symmetry = 'tetragonal-2', equals = tetra2_equal, crosses = tetra2_cross, parities = tetra2_parity)
check_sym(symmetry = 'trigonal-1', equals = trig1_equal, crosses = trig1_cross, parities = trig1_parity)
check_sym(symmetry = 'trigonal-2', equals = trig2_equal, crosses = trig2_cross, parities = trig2_parity)
#check_sym(symmetry = 'hexagonal', subsets = hex_subsets)
#check_sym(symmetry = 'orthorhombic', subsets = ortho_subsets)
#check_sym(symmetry = 'monoclinic', subsets = monoclinic_subsets)
#check_sym(symmetry = 'tetragonal-1', subsets = tetra1_subsets)
#check_sym(symmetry = 'tetragonal-2', subsets = tetra2_subsets)
#check_sym(symmetry = 'trigonal-1', subsets = trig1_subsets)
#check_sym(symmetry = 'trigonal-2', subsets = trig2_subsets)