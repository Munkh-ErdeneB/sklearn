#Author: Munkh-Erdene Baatarsuren 
#2018

import numpy as np
import kaggle
from sklearn.model_selection import cross_val_score 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
import time 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVR 
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LassoCV 
from sklearn.feature_selection import SelectFromModel 

def compute_error(y_hat, y):
    return np.abs(y_hat - y).mean()

def decisionTree(X, y, kaggle_data):
    """
    This function trains decision tree regressor models with different max depths
    Using cross validation with MAE as a locall loss function, this method 
    measures the time each model uses for cross validation and its accuracy from it.
    At the end the plot is made and saved in Figures folder. 
    
    X: Data vectors for training the models  
    y: Labels mapping to each vector in X  
    kaggle_data: data to make prediction for kaggle competition

    """
    print("--------- Decision Tree -----------") 
    #Splitting the data into test and train sets with ratio (80:20) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    
    #Cross validation step: 
    depth_list = [3, 6, 9, 12, 15]
    time_list = []
    for depth in depth_list:
        regressor_tree = DecisionTreeRegressor(max_depth = depth, criterion="mae") 
        start = time.time() 
        c_score = cross_val_score(regressor_tree, X_train, y_train, cv=5, scoring="neg_mean_absolute_error") 
        #adding each measured time into the list for plot
        time_list.append((time.time() - start)*1000)
        mean_ = abs(c_score.mean()) 
        print("depth " + str(depth) + " out of sample error" + ": " + str(mean_)) 
    
    #Plotting step: 
    depth_list_numpy = np.array(depth_list) 
    time_list_numpy = np.array(time_list) 
    plot_path = "../Figures/decisionTree.png" 
    plt.plot(depth_list_numpy, time_list_numpy) 
    plt.xlabel("Depth of the regressor tree") 
    plt.ylabel("Cross validation time (milliseconds)") 
    plt.title("Cross validation time on different regressor tree models") 
    plt.grid(True) 
    plt.savefig(plot_path)
    
    
    #training the model with the best performance on 80% of the total data 
    regressor_tree = DecisionTreeRegressor(max_depth=6, criterion="mae") 
    regressor_tree.fit(X_train, y_train) 
    #Making prediction on the test set, 20% of the total data 
    predict_result = regressor_tree.predict(X_test) 
    MAE_from_test = compute_error(predict_result, y_test)
    print("MAE DT: " + str(MAE_from_test))
    """
    #for kagggle 
    kaggle_dt = DecisionTreeRegressor(max_depth=6, criterion="mae")
    kaggle_dt.fit(X, y) 
    predicted_y = kaggle_dt.predict(kaggle_data)
    # Output file location
    file_name = '../Predictions/best.csv'
    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    """

def KNN(X, y, kaggle_data):
    """
    This function is for training and testing various KNN models, 
    using cross validation technique 

    X: Data vectors for training the models 
    y: Labels mapping to each vector in X. 
    kaggle_data: data to make prediction for kaggle competition
    """
    print("-------------- KNN -------------------") 
    #Splitting the data into test and train sets with ratio (80:20) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    #cross validation on each model 
    neigh_list = [3, 5, 10, 20, 25]
    for neigh in neigh_list: 
        neighbor_regressor = KNeighborsRegressor(n_neighbors = neigh)
        c_score = cross_val_score(neighbor_regressor, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
        mean_ = abs(c_score.mean())
        print("neighbor " + str(neigh) + " out of sample error: " + str(mean_))
    
    #Training the KNN model with the best performance on 80% of the total data 
    regressor_KNN = KNeighborsRegressor(n_neighbors=5) 
    regressor_KNN.fit(X_train, y_train) 
    #Making prediction on the test set, 20% of the total data 
    predict_result = regressor_KNN.predict(X_test) 
    MAE_from_test = compute_error(predict_result, y_test)
    print("MAE KNN: " + str(MAE_from_test))
    #Best model with different distance functions 
    regressor_KNN_minkowski = KNeighborsRegressor(n_neighbors=5, metric="minkowski")
    regressor_KNN_manhattan = KNeighborsRegressor(n_neighbors=5, metric="manhattan")
    regressor_KNN_minkowski.fit(X_train, y_train)
    regressor_KNN_manhattan.fit(X_train, y_train) 
    result_minkowski = regressor_KNN_minkowski.predict(X_test) 
    result_manhattan = regressor_KNN_manhattan.predict(X_test) 
    MAE_mink = compute_error(result_minkowski, y_test) 
    MAE_manh = compute_error(result_manhattan, y_test) 
    print("MAE minkowski: " + str(MAE_mink))
    print("MAE manhattan: " + str(MAE_manh))
    """
    #for kagggle 
    kaggle_knn = KNeighborsRegressor(n_neighbors=5, metric="manhattan")
    kaggle_knn.fit(X, y) 
    predicted_y = kaggle_knn.predict(kaggle_data)
    # Output file location
    file_name = '../Predictions/best.csv'
    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    """
def lin_model(X, y, kaggle_data): 
    """
    This function is for training and testing various Lasso and Ridge regressors,
    using cross validation technique 

    X: Data vectors for training the Models
    y: Labels mapping to each vector in X. 
    kaggle_data: data to make prediction for kaggle competition
    """
    print("--------------- Linear Models ---------------") 
    #Splitting the data into test and train sets with ratio (80:20) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    #normalizing the data 
    normalizer = StandardScaler().fit(X_train) 
    X_train_normalized = normalizer.transform(X_train) 
    X_test_normalized = normalizer.transform(X_test) 

    #Cross validation process 
    alpha_list = [pow(10, -6), pow(10, -4), pow(10, -2), 1, 10]
    for alpha_ in alpha_list:
        ridge_reg = linear_model.Ridge(alpha = alpha_, tol=1) 
        lasso_reg = linear_model.Lasso(alpha = alpha_, tol=1)  
        c_score_ridge = cross_val_score(ridge_reg, X_train_normalized, y_train, cv=5, scoring="neg_mean_absolute_error")
        c_score_lasso = cross_val_score(lasso_reg, X_train_normalized, y_train, cv=5, scoring="neg_mean_absolute_error") 
        mean_ridge = abs(c_score_ridge.mean()) 
        mean_lasso = abs(c_score_lasso.mean()) 
        print("alpha = " + str(alpha_) + " out of sample error --> " + "ridge: " + str(mean_ridge) + ", " + "lasso: " + str(mean_lasso)) 
    
    #training the best performing models on the whole training set 80% of the data 
    lasso_regressor = linear_model.Lasso(alpha=10, tol=1) 
    lasso_regressor.fit(X_train_normalized, y_train) 
    #Making prediction on the test set, 20% of the total data 
    predict_result_lasso = lasso_regressor.predict(X_test_normalized) 
    MAE_lasso = compute_error(predict_result_lasso, y_test) 
    print("MAE_lasso: " + str(MAE_lasso))
    """    
    #for kagggle 
    nor = StandardScaler().fit(X)
    X = nor.transform(X)
    kaggle_data = nor.transform(kaggle_data)
    kaggle_lasso = linear_model.Lasso(alpha=10, tol=1)
    kaggle_lasso.fit(X, y) 
    predicted_y = kaggle_lasso.predict(kaggle_data)
    # Output file location
    file_name = '../Predictions/best.csv'
    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    """
def SVM_reg(X, y, kaggle_data):
    """
    This function is for training and testing various SVM models, 
    using cross validation technique

    X: data vectors for training the models 
    y: Labels mapping to each vectors in X. 
    kaggle_data: data to make prediction for kaggle competition
    """
    print("--------------- SVM --------------------") 
    #Splitting the data into test and train sets with ratio (80:20) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    #normalizing the data 
    normalizer = StandardScaler().fit(X_train) 
    X_train_normalized = normalizer.transform(X_train) 
    X_test_normalized = normalizer.transform(X_test)     

    #cross validation process 
    svm_rbf = SVR(kernel= 'rbf') 
    svm_1 = SVR(kernel= 'linear') 
    svm_2 = SVR(kernel='poly', degree=2)
    rbf_score = cross_val_score(svm_rbf, X_train_normalized, y_train, cv=5, scoring="neg_mean_absolute_error") 
    svm_1_score = cross_val_score(svm_1, X_train_normalized, y_train, cv=5, scoring="neg_mean_absolute_error") 
    svm_2_score = cross_val_score(svm_2, X_train_normalized, y_train, cv=5, scoring="neg_mean_absolute_error") 
    print("SVM rbf -- out of sample error: " + str(abs(rbf_score.mean())))
    print("SVM degree 1 -- out of sample error: " + str(abs(svm_1_score.mean())))
    print("SVM degree 2 -- out of sample error: " + str(abs(svm_2_score.mean())))
    
    #Training the best performing model on the training set, 80% of the total data  
    svm_regressor = SVR(kernel="linear") 
    svm_regressor.fit(X_train_normalized, y_train) 
    #Making prediction on the test set, 20% of the total data 
    predicted_values_svm = svm_regressor.predict(X_test_normalized)
    MAE_svm = compute_error(predicted_values_svm, y_test) 
    print("MAE svm_linear: " + str(MAE_svm))
    
    #for kagggle 
    nor = StandardScaler().fit(X) 
    X = nor.transform(X) 
    kaggle_data = nor.transform(kaggle_data) 
    kaggle_svm = SVR(kernel="linear")
    kaggle_svm.fit(X, y) 
    predicted_y = kaggle_svm.predict(kaggle_data)
    # Output file location
    file_name = '../Predictions/best.csv'
    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)

def NN(X, y, kaggle_data): 
    """
    This function is for training and testing various neural network regressors,
    using cross validation technique

    X: data vectors for training the models
    y: Labels mapping to each vector in X. 
    kaggle_data: data to make prediction for kaggle competition
    """
    print("--------------- NN ----------------") 
    #Splitting the data into test and train sets with ratio (80:20) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    #cross validation process 
    layer_size_list = [10, 20, 30, 40] 
    for units_ in layer_size_list:
        NN_regress = MLPRegressor(hidden_layer_sizes=(units_,))
        nn_cross_score = cross_val_score(NN_regress, X_train, y_train, cv=5, scoring="neg_mean_absolute_error") 
        print("NN with units: " + str(units_) + " -- out of sample error: " + str(abs(nn_cross_score.mean()))) 
    
    #Training the best performing model on the training set, 80% of the total data 
    nn_regressor = MLPRegressor(hidden_layer_sizes=(10,))
    nn_regressor.fit(X_train, y_train) 
    #prediction on the test set, 20% of the total data 
    predicted_nn_values = nn_regressor.predict(X_test) 
    MAE_nn = compute_error(predicted_nn_values, y_test) 
    print("MAE nn: " + str(MAE_nn))
    
    """
    #for kagggle 
    kaggle_nn = MLPRegressor(hidden_layer_sizes=(10,))
    kaggle_nn.fit(X, y) 
    predicted_y = kaggle_nn.predict(kaggle_data)
    # Output file location
    file_name = '../Predictions/best.csv'
    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    """

############################################################################

#Data is not available for public 
#train_x, train_y, test_x   = read_data_fb()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

decisionTree(train_x, train_y, test_x) 
KNN(train_x, train_y, test_x)
lin_model(train_x, train_y, test_x) 
SVM_reg(train_x, train_y, test_x)
NN(train_x, train_y, test_x)
