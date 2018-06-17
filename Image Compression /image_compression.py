#Author: Munkh-Erdene Baatarsuren 
#2018

import numpy as np
import matplotlib.pyplot as plt 
from scipy import misc
from sklearn.cluster import KMeans, AgglomerativeClustering  
from sklearn.utils import shuffle 
from sklearn.feature_extraction.image import grid_to_graph
import math 

def read_data():
    #provide the path to the data as a parameter 
    data_x = misc.imread('')
    return (data_x)

def kmeans_cluster(image_data):
    """
    K-means clustering function on pixels on the input image 
    experimenting on different number of clusters 
    """
    print("----- K-means running... -----")
    image_data = np.array(image_data, dtype=np.float64)/255
    flattened_image = image_data.ravel().reshape(image_data.shape[0] * image_data.shape[1], image_data.shape[2])
    data_train = shuffle(flattened_image, random_state=0)
    #add original image to 3x3 plot 
    image_path = "../Figures/kmeans.png"
    fig = plt.figure(figsize=(24, 24))
    fig.add_subplot(3, 3, 1) 
    plt.title("Original Image") 
    plt.imshow(image_data)
    #k values 
    k_values = [2, 5, 10, 25, 50, 75, 100, 200] 
    #reconstruction error 
    recon_error_dict = {} 
    #clustering for each k value 
    for k, r in zip(k_values, range(2, 10)):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_train) 
        labels = kmeans.predict(flattened_image)
        #plotting the image 
        result_image = np.zeros((image_data.shape[0], image_data.shape[1], image_data.shape[2]))
        clusters = kmeans.cluster_centers_ 
        index_ = 0 
        recon_error = 0 
        for i in range(image_data.shape[0]):
            for j in range(image_data.shape[1]):
                result_image[i][j] = clusters[labels[index_]]
                index_ += 1 
                recon_error += math.sqrt((pow((image_data[i][j][0] - result_image[i][j][0]), 2) + pow((image_data[i][j][1] - result_image[i][j][1]), 2) + pow((image_data[i][j][2] - result_image[i][j][2]), 2))/3)  
        #save reconstruction error 
        recon_error_dict[k] = recon_error
        fig.add_subplot(3, 3, r) 
        plt.title("k-means construction ("+ str(k) + " clusters)") 
        plt.imshow(result_image) 

    plt.savefig(image_path)
    print("reconstruction error dictionary: ") 
    print(recon_error_dict)
    print("k-means finished..") 

def HAC(image_data):
    """
    Hierarchical agglomerative clustering function on the input image 
    experimenting on different number of clusters 
    """
    print("----- HAC running... -----")
    image_data = np.array(image_data, dtype=np.float64)/255
    flattened_image = image_data.ravel().reshape(image_data.shape[0] * image_data.shape[1], image_data.shape[2])
    data_train = shuffle(flattened_image, random_state=0) 
    #add original image to 3x3 plot 
    image_path = "../Figures/HAC.png"
    fig = plt.figure(figsize=(24, 24))
    fig.add_subplot(3, 3, 1) 
    plt.title("Original Image") 
    plt.imshow(image_data)
    #k values 
    k_values = [2, 5, 10, 25, 50, 75, 100, 200] 
    #reconstruction error 
    recon_error_dict = {} 
    #clustering on each k value 
    for k, r in zip(k_values, range(2, 10)):
        #Clusterer intialization
        hac = AgglomerativeClustering(n_clusters=k) 
        hac.fit(data_train) 
        labels = hac.fit_predict(flattened_image) 
        #Sum up elements in each cluster and number of counts
        #Save them in dictionary using cluster ID as a key 
        label_to_pixel_dict = {}
        recon_error = 0 
        for label, indx in zip(labels, range(labels.size)):
            if label not in label_to_pixel_dict:
                label_to_pixel_dict[label] = [flattened_image[indx], 1]
            else: 
                label_to_pixel_dict[label][0] += flattened_image[indx] 
                label_to_pixel_dict[label][1] += 1 
        #find the average value of the cluster and save it in dictionary 
        final_dict = {} 
        for key in label_to_pixel_dict:
            if key in final_dict:
                print("weird error!!!check final dict")
                return None
            final_dict[key] = label_to_pixel_dict[key][0]/label_to_pixel_dict[key][1] 
        
        #preparing the image 
        result_image = np.zeros((image_data.shape[0], image_data.shape[1], image_data.shape[2]))
        index_ = 0
        for i in range(image_data.shape[0]):
            for j in range(image_data.shape[1]):
                result_image[i][j] = final_dict[labels[index_]]
                index_ += 1 
                recon_error += math.sqrt((pow((image_data[i][j][0] - result_image[i][j][0]), 2) + pow((image_data[i][j][1] - result_image[i][j][1]), 2) + pow((image_data[i][j][2] - result_image[i][j][2]), 2))/3)  
        
        #save reconstruction error 
        recon_error_dict[k] = recon_error
        fig.add_subplot(3, 3, r) 
        plt.title("HAC construction ("+ str(k) + " clusters)") 
        plt.imshow(result_image) 
    
    plt.savefig(image_path)
    print("reconstruction error dictionary: ") 
    print(recon_error_dict)
    print("HAC finished..")
if __name__ == '__main__':
    
    #input data image 
    data_x = read_data()
    #K-means 
    kmeans_cluster(data_x)
    #HAC
    HAC(data_x)





