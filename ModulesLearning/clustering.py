from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from SolarForecasting.ModulesLearning import model as models
from SolarForecasting.ModulesLearning import preprocessing as preprocess

def clustering(X, features_indices_to_cluster_on, n_clusters = 2):
    X_sub = X[:,features_indices_to_cluster_on]
    # cluster_plot(X_sub)

    kmeans = KMeans(n_clusters=n_clusters).fit(X_sub)

    return kmeans


# def cluster_plot(X_sub):
#     pca = PCA(n_components=2)
#     principalComponents = pca.fit_transform(X_sub)
#     PCA_components = pd.DataFrame(principalComponents)
#     #
#     plt.scatter(PCA_components[0], PCA_components[1],alpha=.1, color='black')
#     plt.xlabel('PCA_1')
#     plt.ylabel('PCA_2')
#     plt.savefig("PCA_visualization")

def get_closest_clusters(X, kmeans, features_indices_to_cluster_on):
    X_sub = X[:,features_indices_to_cluster_on]
    pred_cluster_labels = kmeans.predict(X_sub)

    return pred_cluster_labels


def normalizing_per_cluster(X_train, X_valid, X_test, cluster_labels, cluster_labels_valid, cluster_labels_test, folder_saving, model, lead):

    n_clusters = len(set(cluster_labels))

    for i in range(n_clusters):

        X_train_task = X_train[cluster_labels == i]
        X_valid_task = X_valid[cluster_labels_valid == i]
        X_test_task = X_test[cluster_labels_test == i]

        X_train_task, X_valid_task, X_test_task = preprocess.standardize_from_train(X_train_task, X_valid_task, X_test_task, folder_saving, model+"task_"+str(i), lead)
        X_train[cluster_labels == i] = X_train_task
        X_valid[cluster_labels_valid == i] = X_valid_task
        X_test[cluster_labels_test == i] = X_test_task

    return X_train, X_valid, X_test

def train(X_train,y_train, cluster_labels, n_clusters=2):

    # kmeans = clustering(X_train,features_indices_to_cluster_on)
    # cluster_labels = kmeans.labels_
    # cluster_valid_labels = get_closest_clusters(X_valid, kmeans, features_indices_to_cluster_on)

    model_dict = {}

    for i in range(n_clusters):
        X_train_i = X_train[cluster_labels == i]
        y_train_i = y_train[cluster_labels == i]

        print("cluster shape: ",X_train_i.shape,"\n")

        # X_valid_i = X_valid[cluster_valid_labels == i]
        # y_valid_i = y_valid[cluster_valid_labels == i]

        model_i = models.rfGridSearch_model(X_train_i, y_train_i)

        model_dict[i] = model_i

    return model_dict


def cluster_and_predict(X, model_dict, predicted_clusters):

    # predicted_clusters = get_closest_clusters(X, kmeans, features_indices_to_cluster_on)
    y_pred = []
    for i in range(len(X)):
        cluster_label = predicted_clusters[i]
        y_pred.append(model_dict[cluster_label].predict(X[i].reshape(1, -1)))

    return y_pred










