import scipy.io as spio
from glob import glob
from pymatgen import MPRester
m = MPRester('TCfhlU2O1TZcNbM0')

files = glob('*non-training-data.mat')
print files
files = [files[-4]]
print files
for f in files:
	data_pred = spio.loadmat(f)
	for i, d in enumerate(data_pred['mps']):
		data_pred['mps'][i] = d.replace(" ","")
	predK = [];
	predG = [];
	for i, d in enumerate(data_pred['mps']):
		a=m.get_data(d, data_type="pred", prop="elastic_moduli")
		predK.append(a[0]['elastic_moduli']['K'])
		predG.append(a[0]['elastic_moduli']['G'])
	spio.savemat('pred-'+f,mdict={'K':predK,'G':predG})



