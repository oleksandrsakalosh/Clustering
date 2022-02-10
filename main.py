import matplotlib.pyplot as plt
import random
import math
import numpy as np
import timeit

Points = []
Colors = [
    '#050A30', 
    '#3cb44b', 
    '#ffe119', 
    '#4363d8', 
    '#f58231', 
    '#911eb4', 
    '#46f0f0', 
    '#f032e6', 
    '#bcf60c', 
    '#fabebe', 
    '#008080', 
    '#e6beff', 
    '#9a6324', 
    '#fffac8', 
    '#800000', 
    '#aaffc3', 
    '#808000', 
    '#ffd8b1', 
    '#000075', 
    '#808080', 
    '#ffffff', 
    '#000000'
    ]
Max_X = 5000
Min_X = -5000
Max_Y = 5000
Min_Y = -5000

class Algorithm:

    def __init__(self, state, points, clust_count = 20, centroids = None, medoids = None, clusters = None):
        self.state = state
        self.points = points
        self.clust_count = clust_count
        if centroids == None and state == 1:
            self.centroids = self.init_centroids()
        else:
            self.centroids = centroids

        if medoids == None and state == 2:
            self.medoids = self.init_centroids()
        else:
            self.medoids = medoids

        if clusters == None:
            self.clusters = self.div_to_clust()
        else:
            self.clusters = clusters

    def init_centroids(self, map_size = [[-5000, 5000], [-5000, 5000]]):
        #x = np.random.randint(map_size[0][0], map_size[0][1], self.clust_count)
        #y = np.random.randint(map_size[1][0], map_size[1][1], self.clust_count)
        #centroids = []
        #for i in range(self.clust_count):
        #    centroids.append([x[i], y[i]])
        centroids = random.choices(self.points, k=self.clust_count)
        return centroids

    def div_to_clust(self):
        clusters = [[] for x in range(self.clust_count)]
        if self.state == 1:
            centers = self.centroids
        else:
            centers = self.medoids
        for point in self.points:
            min = float('inf')
            idx = 0
            for center in centers:
                dist = abs(point[0] - center[0]) + abs(point[1] - center[1])
                if dist < min:
                    min = dist
                    nearest_idx = idx
                idx+=1
            clusters[nearest_idx].append(point)

        return clusters

    def calc_centroid(self, clust):
        sum_x = 0
        sum_y = 0
        count = 0
        for point in clust:
            sum_x += point[0]
            sum_y += point[1]
            count+=1
        
        centroid = [sum_x/count, sum_y/count]
        return centroid

    def calc_medoid(self, clust):
        min_cost = float('inf')
        new_medoid = None
        idx1 = 0
        for point1 in clust:
            idx2 = idx1 + 1
            cost = 0
            for point2 in clust[idx2:]:
                x = abs(point1[0] - point2[0])
                y = abs(point1[1] - point2[1])
                cost += (x + y)
            if cost < min_cost:
                new_medoid = point1
                min_cost = cost
            idx1+=1
        return new_medoid

    def create_matrixofneighbors(self):
        matrix = []
        idx1 = 0
        for center1 in self.centroids:
            row = []
            idx2 = idx1 + 1
            for center2 in self.centroids[idx2:]:
                x = abs(center1[0] - center2[0])
                y = abs(center1[1] - center2[1])
                dist = (x + y)
                row.append(dist)
                idx2+=1
            matrix.append(row)
            idx1+=1
        return matrix

    def update_matrix(self, matrix, idx1, idx2):
        matrix.pop(idx2)
        idx3 = 0
        for row in matrix[:idx2]:
            row.pop(idx2 - idx3 - 1)
            idx3+=1
        idx3 = 0
        for row in matrix[:idx1]:
            center1 = self.centroids[idx3]
            center2 = self.centroids[idx1]
            x = abs(center1[0] - center2[0])
            y = abs(center1[1] - center2[1])
            row[idx1 - idx3 - 1] = x + y
            idx3+=1
        center1=self.centroids[idx1]
        matrix[idx1]=[]
        for center2 in self.centroids[idx1+1:]:
            x = abs(center1[0] - center2[0])
            y = abs(center1[1] - center2[1])
            dist = (x + y)
            matrix[idx1].append(dist)
        return matrix

    def find_closest(self, matrix):
        min_distance = float('inf')
        closest_idx = []
        idx1 = 0
        for row in matrix:
            idx2 = 0
            for dist in row:
                if dist < min_distance:
                    closest_idx = [idx1, idx1 + idx2 + 1]
                    min_distance = dist
                idx2+=1
            idx1+=1
       
        return closest_idx[0], closest_idx[1], matrix

    def kmeans(self):
        new_centroids = []
        for clust in self.clusters:
            if clust != []:
                new_centroids.append(self.calc_centroid(clust))
        self.centroids = new_centroids
        self.clusters = self.div_to_clust()

    def kmedoids(self):
        new_medoids = []
        for clust in self.clusters:
            if clust != []:
                new_medoids.append(self.calc_medoid(clust))
        self.medoids = new_medoids
        self.clusters = self.div_to_clust()

    def agglomerative(self):
        matrix = self.create_matrixofneighbors()
        while self.clust_count > 20:
            idx1, idx2, matrix = self.find_closest(matrix)
            self.clusters[idx1] += self.clusters[idx2]
            self.clusters.pop(idx2)
            self.clust_count=len(self.clusters)
            self.centroids[idx1] = self.calc_centroid(self.clusters[idx1])
            self.centroids.pop(idx2)
            matrix = self.update_matrix(matrix, idx1, idx2)

    def find_next(self, clusters, centroids):
        idx = 0
        max_distance = 0
        next_clust = None
        for clust in clusters:
            distance = 0
            center = centroids[idx]
            for point in clust:
                distance += (abs(point[0] - center[0]) + abs(point[1] - center[1]))
            distance /= len(point)
            if distance > max_distance:
                next_clust = idx
                max_distance = distance
            idx+=1
        return next_clust

if __name__ == '__main__':
    first_points = []

    for i in range(20):
        x = random.randint(Min_X, Max_X)
        y = random.randint(Min_Y, Max_Y)
        point = [x, y]
        Points.append(point)
        first_points.append(point)

    for i in range(20000):
        x, y = random.choice(Points)
        x_offset = random.randint(-100, 100)
        y_offset = random.randint(-100, 100)
        if x + x_offset < Max_X:
            if x + x_offset > Min_X:
                x = x + x_offset
            else:
                x = Min_X
        else:
            x = Max_X
        if y + y_offset < Max_Y:
            if y + y_offset > Min_Y:
                y = y + y_offset
            else:
                y = Min_Y
        else:
            y = Max_Y

        point = [x, y]
        Points.append(point)

    algorithm = input('1. K-means, where the center is the centroid\n2. K-means, where the center is the medoid\n3. Agglomerative clustering, where the center is the centroid\n4. Divisional clustering, where the center is the centroid\n')

    startTime = timeit.default_timer()
    if algorithm == '1':
        km = Algorithm(1, Points)

        for i in range(9):
            km.kmeans()

        endTime = timeit.default_timer()

        centroids = np.array(km.centroids)    
        c=1
        for arr in km.clusters:
            if np.any(arr):
                colors = []
                for a in arr:
                    colors.append(Colors[c-1])
                arr = np.array(arr)
                plt.scatter(arr[:, 0], arr[:, 1], s = 1, c = colors)
                c+=1
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=10, c='r')

        i = 0
        success = []
        for clust in km.clusters:
            if np.any(clust):
                dist = 0
                centroid = km.centroids[i]
                for point in clust:
                    dist += math.sqrt((point[0]-centroid[0])**2+(point[1]-centroid[1])**2)
                dist /= len(clust)
                if dist <= 500:
                    success.append(1)
                else:
                    success.append(0)
                i+=1

        success_rate = 0
        for s in success:
            success_rate += s
        success_rate = success_rate * 100 / len(success)

    if algorithm == '2':
        km = Algorithm(2, Points)

        for i in range(6):
            km.kmedoids()

        endTime = timeit.default_timer()

        medoids = np.array(km.medoids)
    
        c=1
        for arr in km.clusters:
            if np.any(arr):
                colors = []
                for a in arr:
                    colors.append(Colors[c-1])
                arr = np.array(arr)
                plt.scatter(arr[:, 0], arr[:, 1], s = 1, c = colors)
                c+=1
        plt.scatter(medoids[:, 0], medoids[:, 1], marker='*', s=10, c='r')

        i = 0
        success = []
        for clust in km.clusters:
            if np.any(clust):
                dist = 0
                centroid = km.medoids[i]
                for point in clust:
                    dist += math.sqrt((point[0]-centroid[0])**2+(point[1]-centroid[1])**2)
                dist /= len(clust)
                if dist <= 500:
                    success.append(1)
                else:
                    success.append(0)
                i+=1

        success_rate = 0
        for s in success:
            success_rate += s
        success_rate = success_rate * 100 / len(success)

    if algorithm == '3':
        aggl = Algorithm(None, Points, clust_count = len(Points), centroids = Points, clusters = [[Points[i]] for i in range(len(Points))])

        aggl.agglomerative()

        endTime = timeit.default_timer()

        centroids = np.array(aggl.centroids)    
        c=1
        for arr in aggl.clusters:
            if np.any(arr):
                colors = []
                for a in arr:
                    colors.append(Colors[c-1])
                arr = np.array(arr)
                plt.scatter(arr[:, 0], arr[:, 1], s = 1, c = colors)
                c+=1
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=10, c='r')

        i = 0
        success = []
        for clust in aggl.clusters:
            if np.any(clust):
                dist = 0
                centroid = aggl.centroids[i]
                for point in clust:
                    dist += math.sqrt((point[0]-centroid[0])**2+(point[1]-centroid[1])**2)
                dist /= len(clust)
                if dist <= 500:
                    success.append(1)
                else:
                    success.append(0)
                i+=1

        success_rate = 0
        for s in success:
            success_rate += s
        success_rate = success_rate * 100 / len(success)

    if algorithm == '4':
        next = Points
        clusters = []
        centers =[]

        for i in range(19):
            km = Algorithm(1, next, clust_count = 2)

            for i1 in range(5):
                km.kmeans()

            for clust in km.clusters:
                clusters.append(clust)
            for center in km.centroids:
                centers.append(center)

            centroids = np.array(centers)    
            c=1
            if i == 18:
                endTime = timeit.default_timer()
                for arr in clusters:
                    if np.any(arr):
                        colors = []
                        for a in arr:
                            colors.append(Colors[c-1])
                        arr = np.array(arr)
                        plt.scatter(arr[:, 0], arr[:, 1], s = 1, c = colors)
                        c+=1
                plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=10,
                        c='r')
                i1 = 0
                success = []
                for clust in clusters:
                    if np.any(clust):
                        dist = 0
                        centroid = centers[i1]
                        for point in clust:
                            dist += math.sqrt((point[0]-centroid[0])**2+(point[1]-centroid[1])**2)
                        dist /= len(clust)
                        if dist <= 500:
                            success.append(1)
                        else:
                            success.append(0)
                        i1+=1

                success_rate = 0
                for s in success:
                    success_rate += s
                success_rate = success_rate * 100 / len(success)
            else:
                next_idx = km.find_next(clusters, centers)
                next = clusters[next_idx]
                clusters.pop(next_idx)
                centers.pop(next_idx)

    print("Done! Time taken: {}s".format(endTime - startTime))
    print("Success rate: {}%".format(success_rate))
    plt.show()