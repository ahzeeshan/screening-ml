import json 

with open('Li-cmpd-data-Dec18.json') as f:
    data = json.load(f)

size_data = len(data)
index = 0
training_index_mat = []
data_training = []
data_non_training = []
#Finding out the indices of the input data set  
while index<size_data:
    #Segregating the training and non training set here 
    if data[index]['elasticity']!= None:
        training_index_mat.append(index)    
        data_training.append(data[index])
    else:
        data_non_training.append(data[index])         
    index = index+1
# Dumping the segregated training data and non training data into the json file
with open('training-set-data.json','w') as td:
    json.dump(data_training,td) 
with open('non-training-set-data.json','w') as ntd:
    json.dump(data_non_training,ntd)
# Doing random check 


print len(data_training)
print len(data_non_training)