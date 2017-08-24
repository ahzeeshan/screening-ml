import json
import pickle
#test
with open('Li-cmpd-data.json') as f:
    data = json.load(f)

print data[1]

#with open('Li-cmpd-strc-data.pickle') as f:
#    strucdata = pickle.load(f)

#print len(strucdata)
#print strucdata[0]
