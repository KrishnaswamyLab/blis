import networkx as nx
import numpy as np
import random
from blis import DATA_DIR
import os

from sklearn.neighbors import kneighbors_graph

def gaussian(cx, cy, sdx, sdy, locx, locy):
  # centers and loc are tuples
  return np.exp( -((cx - locx)**2/(2 * sdx**2) + (cy- locy)**2 / (2 * sdy**2)) )

# x,y in [0,1]^2, samples f(I x I) -> R
def add_gaussian(vertices, seed):
    np.random.seed(seed)
    centerX1, centerY1, centerX2, centerY2 = np.random.rand(4)
    var = 0.07
    x, y = vertices[:,0], vertices[:,1]
    return gaussian(centerX1, centerY1, var, var, x, y) + gaussian(centerX2, centerY2, var, var, x, y) 

def add_gaussian_diff_height(vertices, seed):
    np.random.seed(seed)
    centerX1, centerY1, centerX2, centerY2 = np.random.rand(4)
    var1, var2 = 0.2, 0.05
    x, y = vertices[:,0], vertices[:,1]
    return gaussian(centerX1, centerY1, var1, var1, x, y) + HEIGHT * gaussian(centerX2, centerY2, var2, var2, x, y) 

def add_gaussian_camel(vertices, seed):
    np.random.seed(seed)
    centerX1, centerY1, centerX2, centerY2 = np.random.rand(4)
    var = 0.2
    shift = 0.3
    x, y = vertices[:,0], vertices[:,1]
    return gaussian(centerX1, centerY1, var, var, x, y) + gaussian((centerX1 + shift), (centerY1 + shift), var, var, x, y) + gaussian(centerX2, centerY2, var, var, x, y) 

def subtract_gaussian(vertices, seed):
    np.random.seed(seed)
    centerX1, centerY1, centerX2, centerY2 = np.random.rand(4)
    var = 0.07
    x, y = vertices[:,0], vertices[:,1]
    return gaussian(centerX1, centerY1, var, var, x, y) - gaussian(centerX2, centerY2, var, var, x, y) 

def add_gaussian_true_camel(vertices, seed):
    np.random.seed(seed)
    centerX1, centerY1 = np.random.rand(2)
    var = 0.07
    x, y = vertices[:,0], vertices[:,1]
    return gaussian(centerX1, centerY1, var, var, x, y) + gaussian(centerX1, centerY1, var/2, var/2, x, y) 

def subtract_gaussian_true_camel(vertices, seed):
    np.random.seed(seed)
    centerX1, centerY1 = np.random.rand(2)
    var = 0.07
    x, y = vertices[:,0], vertices[:,1]
    return gaussian(centerX1, centerY1, var, var, x, y) - gaussian(centerX1, centerY1, var/2, var/2, x, y) 

def gen_vertices(num_nodes):
    # return the number of 
    return np.ndarray.round(np.random.rand(num_nodes, 2), 2)

def make_graph_KNN(vertices, num_neighbors):
    A = kneighbors_graph(vertices, num_neighbors, mode='connectivity', include_self=False)
    G = nx.from_numpy_array(A)
    return G 


class Graph():
    '''
    Given a graph, we would like to generate 
    multiple signals using different functions 
    (hopefully, distinguishing between classes of signals)
    '''
    def __init__(self, G: nx.graph.Graph,
                 num_vertices: int,  
                 vertices: np.array,
                 functions: list, 
                 signals_per_func: int) -> None:
        
        self.graph = G
        self.vertices = vertices 
        self.num_vertices = num_vertices
        self.functions = functions
        self.num_functions = len(functions) 
        self.signals_per_func = signals_per_func 
        self.signals = np.zeros((self.num_functions, self.signals_per_func, self.num_vertices))
        self.labels = np.zeros((self.num_functions, self.signals_per_func))

        # assert( self.num_functions == len(signals_per_func) )

    def gen_signals(self):
        for func_ind, func in enumerate(self.functions):
            for sig_ind in range(self.signals_per_func):
                seed = sig_ind  # generate a seed -- seed could determine the locations of the gaussians, or something
                self.signals[func_ind, sig_ind, :] = func(self.vertices, seed)
                self.labels[func_ind, sig_ind] = func_ind
        return 0
    


if __name__ == "__main__":
    for graph_num in range(5):
        NUM_NODES = 100
        LARGEST_SCALE = 4
        NUM_LAYERS = 3
        DECIMAL = 2
        K = 5   # k-nearest neighbors to create the graph 
        SIGMA = 0.25
        NUM_SIGNALS_TO_GEN = 200
        TEST_SIZE = 0.60

        # generate an array of [0,1]^2
        vertices = gen_vertices(NUM_NODES)
        # use K-nearest-neighbors to generate an adjacency matrix and in turn, a graph
        sparseA = kneighbors_graph(vertices, K, mode='connectivity', include_self=False)
        A = np.array(sparseA.todense())
        G = nx.from_numpy_array(A)

        functions = [add_gaussian, subtract_gaussian, add_gaussian_true_camel, subtract_gaussian_true_camel]
        function_names = ["gaussian_plus", "gaussian_minus", "camel_plus", "camel_minus"]
        function_class = ["gaussian_pm", "camel_pm"]
        graph = Graph(G, NUM_NODES, vertices, functions, NUM_SIGNALS_TO_GEN)
        graph.gen_signals()
        # write to data_dir
        #graph_data_path = os.path.join(DATA_DIR, f"graph_{graph_num}")

        for ind, name in enumerate(function_class):
            # need to save this to the data directory
            graph_data_path = os.path.join(DATA_DIR, f"synthetic/{name}_{graph_num}")
            # check if path exists or not and then do something else with it
            if not os.path.isdir(graph_data_path):
                os.mkdir(graph_data_path) 
            label_dir = os.path.join(graph_data_path, "PLUSMINUS")
            os.mkdir(label_dir)
            # save the adjacency matrix
            graph_signals = graph.signals[ind * 2:(ind * 2 + 2), :, :].reshape(-1,100)
            # save label mod 2 
            labels = graph.labels[ind*2:(ind*2 + 2)].reshape(-1)%2

            np.save(os.path.join(graph_data_path, 'graph_signals.npy'), graph_signals)
            np.save(os.path.join(graph_data_path, 'adjacency_matrix.npy'), A) 
            np.save(os.path.join(label_dir, 'label.npy'), labels)


            

    

    

        

