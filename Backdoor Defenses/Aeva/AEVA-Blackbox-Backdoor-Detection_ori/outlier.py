import numpy as np
import os
import scipy.stats as sta

def sort(array):

    a=np.empty(array.shape[0])

    for i in range(array.shape[0]):

        a[i]=np.sum(np.sort(np.sum(array[i],axis=-1).flatten())[-2:])/np.sum(array[i])
        #a[i] = np.linalg.norm(array[i]) / np.linalg.norm(array[i])

    return a



def get_s(num_labels):
    s=np.zeros(num_labels)
    coll=np.zeros((num_labels,num_labels-1))
    for i in range(0, num_labels):

        labels = list(range(num_labels))
        labels.remove(i)

        assert len(labels) == (num_labels - 1), print(f"Wrong")

        for index, t in enumerate(labels):

            if os.path.exists(f"cifar10_adv_per/data_{t}_{i}.npy"):
                a=np.abs(np.load(f"cifar10_adv_per/data_{t}_{i}.npy"))
                print("cifar_adv_per10/data_{t}_{i}.npy")
                print(a)
                #v=np.max(sort(a))
                v=np.max(sort(a))
                coll[i][index]=v
                
                s[i]+=v
    print(s)
    return s

def analomy(array):

    consistency_constant = 1.4826  # if normal distribution
    median = np.median(array)
    mad = consistency_constant * np.median(np.abs(array - median))
    min_mad = (array - median) / mad
    print(f"Anomaly Index:{min_mad}")

s=get_s(10)
# mad=sta.median_abs_deviation(s)
# print((s-np.median(s))/mad)

analomy(array=s)



