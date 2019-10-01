import numpy as np

def lab_sort(arr, aa):
    sortt = []
    for j in range(0, aa):
        for i in range(arr.shape[0]):
            if(int(j) == int(arr[i][0])):            
                sortt.append(arr[i])
    return np.array(sortt)

def f_sort(arr1, aa):
    final_sort = []
#    cc = []
    for i in range(0,aa):
       for_s = []
       for j in range(arr1.shape[0]):
           if(i == arr1[j][0]):
               for_s.append(arr1[j][-2])
           for_ss  = np.array(for_s)
           sortted_for_s = np.sort(for_ss)
       for k in range(sortted_for_s.shape[0]):
           count = 0 
           for l in range(arr1.shape[0]):                       
              if(arr1[l][-2] == sortted_for_s[k]) and count<1:
                    final_sort.append(arr1[l])
                    count = count+1         #using count so that redundant sum values don't get repeated 
    return np.array(final_sort)

