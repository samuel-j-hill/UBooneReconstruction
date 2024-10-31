import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
import networkx as nx
from sklearn.cluster import DBSCAN
 

class HitsContainer:
    
    def __init__(self, dense_array):
        
        self.dense_array = dense_array
        self.hit_coords = self.get_hit_coords()
        self.clusters_dictionary = None
        
    def get_hit_coords(self):
        
        sparse_array = csr_matrix(self.dense_array)
        x_coords, y_coords = sparse_array.nonzero()
        hit_coords = np.array([[x_coords[i],y_coords[i]] for i in range(sparse_array.getnnz())])
        
        return hit_coords
    
    def make_clusters(self, eps, min_samples):
                
        clustering = DBSCAN(eps=3, min_samples=10).fit(self.hit_coords)
        
        unique_labels = np.unique(clustering.labels_)
        grouped_coords = {label: self.hit_coords[clustering.labels_ == label] for label in unique_labels}
    
        self.clusters_dictionary = {label: Cluster(coords) for label, coords in grouped_coords.items()}
        
        return None 
    
    def plot_clusters(self, color_list):
        
        for label, cluster in self.clusters_dictionary.items():
            cluster.plot_cluster(color_list[label])
        plt.show()
            
        
            

class Cluster:
    
    def __init__(self, hit_coords):
        
        self.hit_coords = hit_coords
        self.endpoints = self.get_endpoints()
        
    def plot_cluster(self, color, show_endpoints=False):
        
        x_values = []
        y_values = []
        for hit_coord in self.hit_coords:
            x_values.append(hit_coord[0])
            y_values.append(hit_coord[1])
            
        plt.scatter(np.array(x_values), np.array(y_values), s=0.1, color=color)
        
        if show_endpoints:
            
            endpoint_1, endpoint_2 = self.get_endpoints()
            plt.scatter(endpoint_1[0], endpoint_1[1], s=1, marker="x")
            plt.scatter(endpoint_2[0], endpoint_2[1], s=1, marker="x")
        
    def get_endpoints(self):
        
        coords = self.hit_coords
        dist_matrix = distance_matrix(coords,coords)
        
        G = nx.Graph()
        
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                G.add_edge(i, j, weight=dist_matrix[i, j])
                
        mst = nx.minimum_spanning_tree(G)
        
        lengths = dict(nx.all_pairs_dijkstra_path_length(mst))
        max_length = 0
        endpoints = [0, 0]

        for u in lengths:
            for v in lengths[u]:
                if lengths[u][v] > max_length:
                    max_length = lengths[u][v]
                    endpoints = [u, v]

        start_node = endpoints[0]
        end_node = endpoints[1]
        
        return ((coords[start_node], coords[end_node]))
    
    def compute_directions(self, end_radius):
        
        pass
    
    def is_aligned_with(self, other_cluster, angle_threshold):
        
        pass
    
    
def calculate_luminance(color):
    
    r, g, b = mcolors.to_rgb(color)
    
    return 0.299 * r + 0.587 * g + 0.114 * b