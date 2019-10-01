# Unsupervised-Image-Forgery-Detection-Using-Discrete-Cosine-Transform-with-K-means-clustering
To detect the forgery in digital images, we are proposing a method for the same. The basic methodology and steps in our proposed method are as follows : (1) first of all, if the image is color (RGB) then we are converting the image to gray-scale, (2) In next step, we are dividing the converted image to many overlapping blocks, in our method the size of these blocks is taken as 8x8, (3) Now, we are applying the DCT on all the blocks in order to extract the features from the blocks and storing them in a matrix, (4) now clustering all the blocks using the k-means algorithm in order to classify or differentiate the extracted features from different blocks, (5) in last step, some lexicographical sorting (here, radix sort) is done on the clusters to sort them in order of most matching to least matching. Finally, In the end of this thesis, results produced are shown in terms of different parameters such as: (i) type of morphological operation on image, (ii) number of clusters used, (iii) correlation limit, (iv) distance limit.


























