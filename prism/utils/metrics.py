import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, v_measure_score, normalized_mutual_info_score
from igraph import Graph
import networkx as nx
from community import community_louvain

class Metrics(object):
    def __init__(self, inference_df):
        self.z, self.labels = self.process(inference_df)
        self.adjacency_matrix = self.get_knn_adjacency(self.z)
        self.graph = Graph.Adjacency((self.adjacency_matrix > 0).tolist())

        self.partition = community_louvain.best_partition(nx.Graph(self.adjacency_matrix), resolution=1.0)
        self.euc_predicted_labels = [self.partition.get(node) for node in range(len(self.z))]
        
        self.cosine_sim, self.vectors = self.get_cosine_similarity(self.z)
        self.cos_predicted_labels, self.cos_predicted_labels_res_07 = self.cosine_louvain(self.vectors, self.cosine_sim)
        
        self.ari_score_euc = adjusted_rand_score(self.labels, self.euc_predicted_labels)
        self.ari_score_cos = adjusted_rand_score(self.labels, self.cos_predicted_labels)
        self.ari_score_cos_07 = adjusted_rand_score(self.labels, self.cos_predicted_labels_res_07)

        self.v_measure = v_measure_score(self.labels, self.cos_predicted_labels)
        self.nmi = normalized_mutual_info_score(self.labels, self.cos_predicted_labels)
        self.nmi_07 = normalized_mutual_info_score(self.labels, self.cos_predicted_labels_res_07)
        
    def process(self, df):
        data = df.iloc[:, 0:-1]  # Assuming the last column is labels
        labels = df.iloc[:, -1].to_list()
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data, labels
    
    def get_knn_adjacency(self, data, k=15):
        data = torch.from_numpy(data).to("cuda:1")
        dist_matrix = self.get_pw_dist(data)
        dist_matrix = dist_matrix.cpu().numpy()
        adjacency_matrix = np.zeros(dist_matrix.shape)
        for i in range(dist_matrix.shape[0]):
            sorted_indices = np.argsort(dist_matrix[i, :])
            adjacency_matrix[i, sorted_indices[0:k+1]] = 1  # Skipping the first one as it is the point itself
        return adjacency_matrix
    
    def get_pw_dist(self, z):
        distance_matrix = torch.cdist(z, z, p=2)
        return distance_matrix
    
    def get_cosine_similarity(self, z):
        vectors = z
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norms
        cosine_similarity = np.dot(normalized_vectors, normalized_vectors.T)
        cosine_similarity = (cosine_similarity + 1) / 2
        return cosine_similarity, vectors
    
    def cosine_louvain(self, vectors, cosine_sims, k=15):
        G = nx.Graph()
        for i in range(len(vectors)):
            G.add_node(i)
            top_k_indices = np.argsort(cosine_sims[i])[-k-1:-1]  # Exclude self-connection
            for j in top_k_indices:
                if i != j:  # Avoid self-loops
                    G.add_edge(i, j, weight=cosine_sims[i][j])
        
        partition = community_louvain.best_partition(G, resolution=1.0)
        partition2 = community_louvain.best_partition(G, resolution=0.7)
        labels = [partition.get(node) for node in range(len(vectors))]
        labels2 = [partition2.get(node) for node in range(len(vectors))]
        return labels, labels2