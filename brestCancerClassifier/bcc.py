import numpy as np
import pandas as pd
import math,random
from collections import Counter


def phc(data,prediction,k=3):
    
    votes = []
    for hand in data:
        for features in data[hand]:
            dist = np.linalg.norm(np.array(hand)-np.array(prediction))
            votes.append([dist,hand])
    
    results = [i[1] for i in sorted(votes)[:k]]
    result = Counter(results).most_common(1)[0][0]
    return result

data = pd.read_csv('breast-cancer-wisconsin.data.txt')

data.replace('?',-99999,inplace=True)
data.drop(['id'],1,inplace=True)

data=data.astype(float).values.tolist()
random.shuffle(data)

test = data[:100]
train = data[100:]
test_data = {2:[],4:[]}
train_data = {2:[],4:[]}


for group in train:
    train_data[group[-1]].append(group[:-1])

for group in test:
    test_data[group[-1]].append(group[:-1])   
successful = 0.0
total = 0.0

for group in test_data:
    for features in test_data[group]:
        klass = phc(train_data,features,k=5)
        if klass == group:
            successful+=1
        total+=1    


print successful/total

print phc(train_data,[2.3,2.6,3.8,4.2,1.3,2.1,1.3,6.5,1.9],5)