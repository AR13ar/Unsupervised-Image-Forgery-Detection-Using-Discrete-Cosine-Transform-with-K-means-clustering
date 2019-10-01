import cv2          
from matplotlib import pyplot as plt  
import numpy as np   
from scipy.fftpack import dct
from sklearn.cluster import KMeans
from scipy.stats import pearsonr   

#reading the image and converting into gray-scale (0-> gray || 1->RGB)
image = cv2.imread('D:\\PROJECTS\\pytorch_github\\project_2\\006_F.png',0)


#resizing the image and viewing it
image = cv2.resize(image,None,fx=0.4,fy=0.4)
plt.imshow(image, cmap ='gray')

#Thresholded image to boost accuracy
#ret, image = cv2.threshold(image, 0,255, cv2.THRESH_OTSU)
#cv2.THRESH_BINARY +
#selecting the 16 zig-zag sequence of DCT coefficients
x=[0,0,1,2,1,0,0,1,2,3,4,3,2,1,0,0]
y=[0,1,0,0,1,2,3,2,1,0,0,1,2,3,4,5]
#x = np.random.choice(16,8)
#y = np.random.choice(16,8)

dct_coeff_list_16 = []
dct_list, win_data = dct_block(image)
for i in range(dct_list.shape[0]):
    dct_coeff=[]
    for j in range(len(x)):
        dct_coeff.append(dct_list[i,x[j],y[j]])
    dct_coeff_list_16.append(np.array(dct_coeff))
    
    
center, labels, k_dct = kmeans(dct_coeff_list_16)

#lines 71-77 are just meant to help backtracking by forming a new matrix con_cat by merging labels and K_dct
aa = int(center.shape[0])
ll = np.zeros(aa)
for i,row in enumerate(range(1,aa+1,1)):
    ll[i] = row
ll = ll.reshape(ll.shape[0], 1)    
con_cat = np.concatenate((labels, k_dct), axis = 1)


#fing the sum of the rows i.e sum of features of clusters row-wise to distinguish each row from another as 
# a result to help us sorting the feature vectors

#find the sum of each row i.e sum of 16 coeff values of each block
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
ll_con_cat = np.concatenate((con_cat_1, ll_576), axis = 1)
ll_con_cat_1 = lab_sort(ll_con_cat, aa)
#this function sorts the rows (feature vectors) of a single cluster type at a time according to their
#corresponding normalised sum values. this is done for each group(cluster type)  
ll_con_cat_11 = f_sort(ll_con_cat_1, aa)

#correlation part
#pearsonr is used to find the correlation (same formula as in paper)
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
    if(pear_1[i][0] > 0.99995):
        pear_2[i] = pear_1[i]

#now putting threshold on pear_2 values based on the distance factor & storing in pear_3. Those above threshold are    
#stored in pear_2 list as it is and rest all other are stored as 0(zero) in pear_2 
#this is done so that the dimension remains same and labels remain unaffected
pear_3 = np.zeros((pear_2.shape[0],2))
for i in range(pear_2.shape[0]-1):
    if(pear_2[i][0] != 0) and (pear_2[i+1][1] != 0):
        indx = np.abs(((pear_2[i][1]) - pear_2[i+1][1])/int(pear_2[i][1]))
        if( indx >0.7 and indx < 0.99):         #(pear_1.shape[0]) > 0.40):
            pear_3[i] =  pear_2[i]#np.abs((pear_2[i][1]) - pear_2[i+1][1])/int(pear_1.shape[0])#
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

#making rectangles around the pixels or points in the image that are found after back tracking-
rect=np.zeros_like(image.shape)
rect=image
did = int(np.sqrt(win_data.shape[0]))
for i in range(for_index.shape[0]):
    a1_x=((for_index[i]-1)%did)*5
    a1_y=(int((for_index[i]-1)/did))*5    
    a1_xe=a1_x + 10
    a1_ye=a1_y + 10    
    rect=cv2.rectangle(rect,(a1_x,a1_y),(a1_xe,a1_ye),(0,255,0))  
plt.imshow(rect, cmap = 'gray')
plt.show()

