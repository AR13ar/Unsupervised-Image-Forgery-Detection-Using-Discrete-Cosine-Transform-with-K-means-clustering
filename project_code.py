# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2          #computer vision lib, to perform various operations on image (read, write, etc)
from matplotlib import pyplot as plt  #to plot the results
import numpy as np   #to perform any arithematic operation
from scipy.fftpack import dct
from sklearn.cluster import KMeans


#reading the image and converting into gray-scale (0-> gray || 1->RGB)
image = cv2.imread('C://Users//dell//Desktop//project related//pictures//006_F.png',0)


#resizing the image and viewing it
image=cv2.resize(image,None,fx=0.4,fy=0.4)
plt.imshow(image, cmap ='gray')


#dividing the gray-scale image into overlapping blocks of size 8x8
a,b= image.shape       #image.shape->reading the size of the image
fft_img = []
win_data = []
filt_dim = 8     #filt_dim signifies the size of the window of the filter which will move on the image
stride = 2       #stride means how many steps the filter moves at a time i.e defines the overlapping
for i, row in enumerate(range (0,a-filt_dim,stride)):
    for j, col in enumerate(range(0, b-filt_dim,stride)):
        win = image[row:row+filt_dim,col:col+filt_dim]
        win_data.append(win.flatten())   #append means adding to the list
win_data = np.array(win_data)            #np.array converts the list into 2-D array form


#applying DCT on the image
dct_list=[]
for i in range(win_data.shape[0]):    
    dct_list.append(dct(np.reshape(win_data[i],(filt_dim,filt_dim)), 1))
dct_list=np.array(dct_list)


#selecting the 16 zig-zag sequence of DCT coefficients
x=[0,0,1,2,1,0,0,1,2,3,4,3,2,1,0,0]
y=[0,1,0,0,1,2,3,2,1,0,0,1,2,3,4,5]

dct_coeff_list_16=[]
for i in range(dct_list.shape[0]):
    dct_coeff=[]
    for j in range(len(x)):
        dct_coeff.append(dct_list[i,x[j],y[j]]   )
    dct_coeff_list_16.append(np.array(dct_coeff))
    


#applying KMeans
labels = []       #to store labels of the clusters
center = []       #to store centre of each cluster
k_dct = np.array(dct_coeff_list_16)           #converting 16 coeff list to K_dct array
k_dct = k_dct.reshape(k_dct.shape[0], 16)     #appending the 16 dct coeffs from all the blocks into K_dct array 
k_dct = (k_dct-k_dct.mean())/k_dct.std()
clf=KMeans(50)                               #no. of clusters
clf.fit(k_dct)    
labels.append( clf.labels_)            #finding the labels
center.append(clf.cluster_centers_)    #finding the centres 
center = np.array(center)
center = center.reshape(center.shape[1], center.shape[2])   
labels = np.array(labels)
labels = labels.T                  #taking transpose of the labels matrix


#lines 71-77 are just meant to help backtracking by forming a new matrix con_cat by merging labels and K_dct
aa = int(center.shape[0])
ll = np.zeros(aa)
for i,row in enumerate(range(1,aa+1,1)):
    ll[i] = row
ll = ll.reshape(ll.shape[0], 1)    
con_cat = np.concatenate((labels, k_dct), axis = 1)


#fing the sum of the rows i.e sum of features of clusters row-wise to distinguish each row from another as 
# a result to help us sorting the feature vectors

#fing the sum of each row i.e sum of 16 coeff values of each block
sum_ = []
for i in range(k_dct.shape[0]):
    sum_.append((k_dct[i]).sum()) 
    
sum_1 = np.array(sum_)
sum_1 = sum_1.reshape(sum_1.shape[0],1)
sum_11 = np.concatenate((labels, sum_1), axis = 1)   #attaching labels list to their corresponding to row 
                                                     #sum values to ease the backtracking
#fing the sum of all the centres
sum_center = []
for i in range(center.shape[0]):
    sum_center.append((center[i]).sum())

sum_center = np.array(sum_center)                            #converting from list to matrix
sum_center = sum_center.reshape(sum_center.shape[0],1)
sum_center_11 = np.concatenate((ll,sum_center), axis = 1)    #same what we were doing from line 89-90

group = []
for i in range(sum_11.shape[0]):   
    for j in range(sum_center_11.shape[0]):
        if(sum_11[i][0] == (sum_center_11[j][0] - 1)):
            group.append((sum_center_11[j][1] - sum_11[i][1])/center[j].std() )  #normalization of values
  

#normalizing the value of sum we got coreesponding to each row using sum_centre        
group = np.array(group)
group = group.reshape(group.shape[0], 1)
group_1 = np.concatenate((labels, group), axis = 1)   #attaching labels to normalised vales to ease back-track
bb = int(win_data.shape[0])
ll_576 = np.zeros(bb)
for i,row in enumerate(range(1,bb + 1,1)):
    ll_576[i] = row
ll_576 = ll_576.reshape(ll_576.shape[0], 1)  

group_11 = np.concatenate((group_1, ll_576), axis = 1)
con_cat_1 = np.concatenate((con_cat, group), axis = 1)



#this function sorts the labels of the clusters i.e sorting the matrix based on the values of "label" coloumn
def lab_sort(arr):
    sortt = []
    for j in range(0, aa):
        for i in range(arr.shape[0]):
            if(int(j) == int(arr[i][0])):            
                sortt.append(arr[i])
    return np.array(sortt)
ll_con_cat = np.concatenate((con_cat_1, ll_576), axis = 1)
ll_con_cat_1 = lab_sort(ll_con_cat)



#this function sorts the rows (feature vectors) of a single cluster type at a time according to their
#corresponding normalised sum values. this is done for each group(cluster type)  
def f_sort(arr1):
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

ll_con_cat_11 = f_sort(ll_con_cat_1)




#correlation part
from scipy.stats import pearsonr      #pearsonr is used to find the correlation (same formula as in paper)
corr = ll_con_cat_11[:,1:-2]

#finding the correlation between ith row and (i+1)th row & storing in pear
pear = []
for i in range(corr.shape[0]-1):
    pear.append(pearsonr(corr[i], corr[i+1]))
pear.append(pearsonr(corr[-1], corr[0]))
pear = np.array(pear) 
pear_1 = np.concatenate((pear[:,0].reshape(pear.shape[0], 1),  (ll_con_cat_11[:, (ll_con_cat_11.shape[1]-1)]).reshape(ll_con_cat_11.shape[0],1)), axis = 1) 
#in above line,in pear_1 we concatenated the pear list and labels to ease the back-track 

#now putting a threshold on the correlation values stored in pear. Those above threshold are 
#stored in pear_2 list as it is and rest all other are stored as 0(zero) in pear_2 
#this is done so that the dimension remains same and labels remain unaffected
pear_2 = np.zeros((pear_1.shape[0],2 ))
for i in range(pear_1.shape[0]):
    if(pear_1[i][0]> 0.99995):
        pear_2[i] = pear_1[i]

#now putting threshold on pear_2 values based on the distance factor & storing in pear_3. Those above threshold are    
#stored in pear_2 list as it is and rest all other are stored as 0(zero) in pear_2 
#this is done so that the dimension remains same and labels remain unaffected
pear_3 = np.zeros((pear_2.shape[0],2))
for i in range(pear_2.shape[0]-1):
    if(pear_2[i][0] != 0) and (pear_2[i+1][1] != 0):
        if(np.abs((pear_2[i][1]) - pear_2[i+1][1])/int(pear_1.shape[0]) > 0.40):
            pear_3[i] = pear_2[i]
pear_2 = pear_3


#now backtracking the values found to the actual points or pixels in the actual image
back_track = np.zeros((win_data.shape))
for i in range(pear_2.shape[0]):
    if(pear_2[i][1] != 0 ):
        c = int(pear_2[i][1])-1
        back_track[c] = win_data[c]

for_index = []
for i in range(back_track.shape[0]):
    if(back_track[i][1] != 0):
        for_index.append(i)
for_index = np.array(for_index)              

'''    
from scipy.stats import pearsonr
corr = sort[:,1:-1]
pear = []
for i in range(corr.shape[0]-1):
    pear.append(pearsonr(corr[i], corr[i+1]))
pear.append(pearsonr(corr[-1], corr[0]))
pear = np.array(pear)                 
sort_1 = sort[:,0] 
sort_1 = sort_1.reshape(sort_1.shape[0], 1)
pear_1 = np.concatenate((pear, sort_1 ), axis = 1)             
pear_2 = np.zeros(pear_1.shape[0])
for i in range(pear_1.shape[0]):
    if(pear_1[i][0] >= 0.999999):
        pear_2[i] = pear_1[i][2]
        
pear_2 = pear_2.reshape(pear_2.shape[0],1)
sort_corr_pear = sort[:,-1]
sort_corr_pear = sort_corr_pear.reshape(sort_corr_pear.shape[0], 1)
corr_pear = np.concatenate((pear_2, corr, sort_corr_pear), axis = 1)

ll_con_cat = np.concatenate((ll_576,con_cat_1), axis = 1)
sort_11 = np.sort(ll_con_cat, axis = 0)
sort_11 = sort_11[:,1:sort_11.shape[1]]
sort12 = np.concatenate((pear_2, sort_11), axis = 1)

new_bck = []
for i in range(corr_pear.shape[0]):
    for j in range(con_cat_1.shape[0]):
        if(con_cat_1[i][-1] == corr_pear[j][-1]):
            new_bck.append([corr_pear[j][0], corr_pear[j][-1]])

'''
'''
k_dct = k_dct.reshape(k_dct.shape[0], 16)
labels = np.array(labels)

from scipy.stats import pearsonr

corr1 = []
corr2 = []
for i in range(fft_img.shape[0]):
    for j in range(fft_img.shape[0]):
        if(i != j):
            corr1.append(pearsonr(labels[i], labels[j]))
            corr2.append([i,j])
#        corr2.append(np.corrcoef(labels[i], labels[i+1]))
corr1 = np.array(corr1)   

for_indx = []
for i in range(corr1.shape[0]):
#    for j in range(fft_img.shape[0]):    
        if(np.abs(corr1[i][0]) >= 0.9999):
            for_indx.append(corr2[i])
for_indx = (np.array(set(for_indx))).flatten()
'''
'''
index1 = []
index2 = []
for i in range(for_indx.shape[0]):
    if (for_indx[i][0] != for_indx[i][1] ):
        index1.append(for_indx[i][0])
        index2.append(for_indx[i][1])
'''    


#making rectangles around the pixels or points in the image that are found after back tracking-
rect=np.zeros_like(image.shape)
rect=image
did = int(np.sqrt(win_data.shape[0]))
for i in range(for_index.shape[0]):
    a1_x=((for_index[i]-1)%did)*5
    a1_y=(int((for_index[i]-1)/did))*5    
    a1_xe=a1_x+8
    a1_ye=a1_y+8    
    rect=cv2.rectangle(rect,(a1_x,a1_y),(a1_xe,a1_ye),(0,255,0))  
plt.imshow(rect)
plt.show()










#sum_ = 0
#for i in range(indx.shape[0]):
#    sum_ += (np.sqrt(indx[i][0]**2 - 2*indx[i][0]*indx[i][1] + indx[i][1]**2)) 
#    
    