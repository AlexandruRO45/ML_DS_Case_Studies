#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import pickle
import random
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, v_measure_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

def calculate_percent(sub_df, attrib):
    cnt = sub_df[attrib].count()
    output_sub_df = sub_df.groupby(attrib).count()
    return (output_sub_df / cnt)

# Load Feature Sets
pge_path = 'pge_dim_reduced_feature.h5'
vgg16_path = 'vgg16_dim_reduced_feature.h5'

pge_content = h5py.File(pge_path, mode='r')
vgg16_content = h5py.File(vgg16_path, mode='r')

# PCA feature from 2 feature sets: pge_latent, vgg16_latent
pge_pca_feature = pge_content['pca_feature'][...]
vgg16_pca_feature = vgg16_content['pca_feature'][...]

# UMAP feature from 2 feature sets: pge_latent, vgg16_latent
pge_umap_feature = pge_content['umap_feature'][...]
vgg16_umap_feature = vgg16_content['umap_feature'][...]

# Tissue type as available ground-truth: labels
filename_pge = np.squeeze(pge_content['file_name'])
filename_pge = np.array([str(x) for x in filename_pge])
labels_pge = np.array([x.split('/')[2] for x in filename_pge])

filename_vgg16 = np.squeeze(vgg16_content['file_name'])
filename_vgg16 = np.array([str(x) for x in filename_vgg16])
labels_vgg16 = np.array([x.split('/')[2] for x in filename_vgg16])

random.seed(0)
selected_index_pge = random.sample(list(np.arange(len(pge_pca_feature))), 200)
selected_index_vgg16 = random.sample(list(np.arange(len(vgg16_pca_feature))), 200)

# Selected PCA and UMAP features for PathologyGAN
selected_pca_data_pge = pge_pca_feature[selected_index_pge]
selected_umap_data_pge = pge_umap_feature[selected_index_pge]
selected_labels_pge = labels_pge[selected_index_pge]

# Selected PCA and UMAP features for VGG16
selected_pca_data_vgg16 = vgg16_pca_feature[selected_index_vgg16]
selected_umap_data_vgg16 = vgg16_umap_feature[selected_index_vgg16]
selected_labels_vgg16 = labels_vgg16[selected_index_vgg16]

# Function to calculate silhouette and v-measure scores for a given range of clusters and plot the scores
def calculate_and_plot_scores_range(data, labels, feature_type, dataset):
    silhouette_scores = []
    v_measure_scores = []

    cluster_range = range(2, 10)

    for num_clusters in cluster_range:
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans_assignment = kmeans_model.fit_predict(data)

        silhouette = silhouette_score(data, kmeans_assignment)
        v_measure = v_measure_score(labels, kmeans_assignment)

        silhouette_scores.append(silhouette)
        v_measure_scores.append(v_measure)

        # Plot cluster configuration for each cluster
        resulted_cluster_df = pd.DataFrame({'clusterID': kmeans_assignment, 'type': labels})
        label_proportion_df = resulted_cluster_df.groupby(['clusterID']).apply(
            lambda x: calculate_percent(x, 'type')).rename(
            columns={'clusterID': 'type_occurrence_percentage'}).reset_index()
        pivoted_label_proportion_df = pd.pivot_table(label_proportion_df, index='clusterID', columns='type',
                                                      values='type_occurrence_percentage')

        f, axes = plt.subplots(1, 1, figsize=(10, 5))
        df_idx = pivoted_label_proportion_df.index
        (pivoted_label_proportion_df * 100).loc[df_idx].plot.bar(stacked=True, ax=axes)
        axes.set_ylabel('Percentage of tissue type')
        axes.legend(loc='upper right')
        axes.set_title(f'Cluster configuration by Kmeans for {feature_type} Features - {dataset} (Clusters={num_clusters})')

        plt.show()

        # Display K-Means Assignment Counts Table for each cluster configuration
        kmeans_counts = np.unique(kmeans_assignment, return_counts=True)
        kmeans_assignment_counts = pd.DataFrame({'Dataset': [dataset]*len(kmeans_counts[0]),
                                                 'Feature Type': [feature_type]*len(kmeans_counts[0]),
                                                 'Cluster Index': kmeans_counts[0],
                                                 'Number of members': kmeans_counts[1]})
        print(f"\n=== K-Means Assignment Counts for {feature_type} {dataset} (Clusters={num_clusters}) ===")
        print(kmeans_assignment_counts.set_index(['Dataset', 'Feature Type', 'Cluster Index']))

    # Plot Silhouette and V-Measure Scores
    plt.figure(figsize=(8, 4))
    plt.plot(cluster_range, silhouette_scores, marker='o', label='Silhouette Score')
    plt.plot(cluster_range, v_measure_scores, marker='o', label='V-Measure Score')
    plt.title(f'Silhouette and V-Measure Scores for {feature_type} {dataset}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

    # Find the index of the maximum score for both silhouette and v-measure
    ideal_clusters_silhouette = np.argmax(silhouette_scores) + 2
    ideal_clusters_v_measure = np.argmax(v_measure_scores) + 2

    # Print the ideal number of clusters
    print(f'Ideal number of clusters for {feature_type} {dataset} based on Silhouette Score: {ideal_clusters_silhouette}')
    print(f'Ideal number of clusters for {feature_type} {dataset} based on V-Measure Score: {ideal_clusters_v_measure}')

    return ideal_clusters_silhouette, ideal_clusters_v_measure

# Calculate, plot scores, and print ideal clusters for PCA and UMAP features for both datasets
ideal_clusters_pca_pge = calculate_and_plot_scores_range(selected_pca_data_pge, selected_labels_pge, 'PCA', 'PathologyGAN')
ideal_clusters_umap_pge = calculate_and_plot_scores_range(selected_umap_data_pge, selected_labels_pge, 'UMAP', 'PathologyGAN')
ideal_clusters_pca_vgg16 = calculate_and_plot_scores_range(selected_pca_data_vgg16, selected_labels_vgg16, 'PCA', 'VGG16')
ideal_clusters_umap_vgg16 = calculate_and_plot_scores_range(selected_umap_data_vgg16, selected_labels_vgg16, 'UMAP', 'VGG16')

# Create a table for ideal clusters
clusters_data = {
    'Dataset': ['PathologyGAN', 'PathologyGAN', 'VGG16', 'VGG16'],
    'Feature Type': ['PCA', 'UMAP', 'PCA', 'UMAP'],
    'Ideal Clusters (Silhouette)': [ideal_clusters_pca_pge[0], ideal_clusters_umap_pge[0], ideal_clusters_pca_vgg16[0], ideal_clusters_umap_vgg16[0]],
    'Ideal Clusters (V-Measure)': [ideal_clusters_pca_pge[1], ideal_clusters_umap_pge[1], ideal_clusters_pca_vgg16[1], ideal_clusters_umap_vgg16[1]]
}

clusters_table = pd.DataFrame(clusters_data)

# Display the table in a prettier format with borders
table_str = tabulate(clusters_table, headers='keys', tablefmt='fancy_grid', showindex=False)
print("\n=== Ideal Number of Clusters ===")
print(table_str)


# In[ ]:




