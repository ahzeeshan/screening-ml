import json
import math

with open('cubic-training-set-data.json') as f:
	data = json.load(f)

print len(data)

file = 'non-zeros.txt'
with open(file, 'w') as f:
	f.write('i j mp formula\n')
	'''
cubic_combs = {}
for i, elem in enumerate(data):
	elastic_mat = elem['elasticity']['elastic_tensor']
	assert np.allclose(elastic_mat.transpose() , elastic_mat)
	for j in range(6):
		for k in range(6):
			if math.fabs(elem['elasticity']['elastic_tensor'][j][k]) > 1e-5:
				with open(file,'a') as f:
					f.write('{} {} {} {}\n'.format(j+1,k+1,elem['task_id'],elem['pretty_formula']))
'''

cubic_subsets = {(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(4,4),(5,5),(6,6)}
for index, elem in enumerate(data):
	elastic_mat = elem['elasticity']['elastic_tensor']
	assert np.allclose(elastic_mat.transpose() , elastic_mat)
	for i, elem in enumerate(data):
		for subset in itertools.combinations_with_replacement(a,2):
			print subset
			if math.fabs(elem['elasticity']['elastic_tensor'][subset[0]][subset[1]]) > 1e-5 and subset not in subsets:
				with open(file,'a') as f:
					f.write('{} {} {} {}\n'.format(j+1,k+1,elem['task_id'],elem['pretty_formula']))