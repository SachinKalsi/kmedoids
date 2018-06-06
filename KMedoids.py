from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random
import matplotlib.pyplot as plt

class KMedoids:
    def __init__(self, n_cluster=2, max_iter=50, tol=0.0001):
        '''Kmedoids constructor called'''
        self.n_cluster = n_cluster
        self.tol = tol
        self.max_iter = max_iter
        self.__distance_matrix = None
        self.__data = None
        self.__is_csr = None
        self.__rows = 0
        self.__columns = 0
        self.__distance_matrix = None
        self.medoids = []
        self.clusters = {}
        self.tol_reached = float('inf')
        self.__current_distance = 0
        
    def fit(self, data):
        self.__data = data
        self.__set_data_type()
        self.__start_algo()
    
    def __start_algo(self):
        self.__initialize_medoids()
        self.clusters, self.__current_distance = self.__calculate_clusters(self.medoids)
        self.__update_clusters()
        
    def __update_clusters(self):
        for i in range(self.max_iter):
            if self.tol_reached <= self.tol:
                break
            self.__swap_and_recalculate_clusters()

    def __swap_and_recalculate_clusters(self):
        for row in range(self.__rows):
            if self.tol_reached <= self.tol:
                break
#             if row not in self.medoids:
            self.__swap_and_calculate_new_distance(row)
                
     
    def __swap_and_calculate_new_distance(self, row):
        index = 0
        for medoid in range(self.n_cluster):
            temp_medoids_list = list(self.medoids)
            temp_medoids_list[index] = self.medoids[medoid]
            clusters, distance_ = self.__calculate_clusters(temp_medoids_list)
            self.tol_reached = abs(distance_ - self.__current_distance)
            if distance_ < self.__current_distance:
                self.__current_distance = distance_
                self.clusters = clusters
                self.medoids = temp_medoids_list[:]
                break
            index += 1
        
    def __calculate_clusters(self, medoids):
        clusters = {}
        distance = 0
        for row in range(self.__rows):
#             if row not in medoids:
            nearest_medoid, nearest_distance = self.__get_shortest_distance_to_mediod(row)
            distance += nearest_distance
            if nearest_medoid not in clusters.keys():
                clusters[nearest_medoid] = []
            clusters[nearest_medoid].append(row)
        return clusters, distance
        
        
    def __initialize_medoids(self):
        self.medoids.append(random.randint(0,self.__rows-1))
        while len(self.medoids) != self.n_cluster:
            self.medoids.append(self.__find_distant_medoid())
    
    def __find_distant_medoid(self):
        distances = []
        indices = []
        for row in range(self.__rows):
#             if row not in self.medoids:
            indices.append(row)
            distances.append(self.__get_shortest_distance_to_mediod(row)[1])
        distances_index = np.argsort(distances)
        return indices[self.__get_distant_medoid(distances_index)]
    
    def __get_distant_medoid(self, distances_index):
        start_index = round(0.8*len(distances_index))
        end_index = len(distances_index)-1
        return distances_index[random.randint(start_index, end_index)]
        
    def __get_shortest_distance_to_mediod(self, row_index):
        min_distance = float('inf')
        current_medoid = None
        for medoid in self.medoids:
            current_distance = self.__get_distance(medoid, row_index)
            if current_distance < min_distance:
                min_distance = current_distance
                current_medoid = medoid
        return current_medoid, min_distance
                           
    def __get_distance(self, x1, x2):
        a = self.__data[x1].toarray() if self.__is_csr == True else np.array(self.__data[x1])
        b = self.__data[x2].toarray() if self.__is_csr == True else np.array(self.__data[x2])
        return np.linalg.norm(a-b)
    
    def __set_data_type(self):
        if isinstance(self.__data,csr_matrix):
            self.__is_csr = True
            self.__rows = self.__data.shape[0]
            self.__columns = self.__data.shape[1]
        elif isinstance(self.__data,list):
            self.__is_csr = False
            self.__rows = len(self.__data)
            self.__columns = len(self.__data[0])
        else:
            raise ValueError('Invalid input')
            