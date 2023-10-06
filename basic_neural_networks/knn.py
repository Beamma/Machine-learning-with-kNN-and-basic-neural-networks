import math

def euclidean_distance(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i])**2
    
    return math.sqrt(sum)

def majority_element(labels):

    mode = {}

    for i in range(len(labels)):
        if labels[i] not in mode.keys():
            mode[labels[i]] = 1
        
        else:
            mode[labels[i]] += 1

    
    mode_list = [(v, k) for k, v in mode.items()]
    
    return max(mode_list)[1]

def knn_predict(input, examples, distance, combine, k):
    
    neighbours = []
    examples_copy = examples.copy()

    for i in range(len(examples)):
        dist = distance(input, examples[i][0])
        neighbours.append(dist) 

    og = neighbours.copy()

    k_neighbours = []
    for i in range(k):
        smallest = min(neighbours)
        k_neighbours.append(smallest)
        neighbours[neighbours.index(smallest)] = math.inf

    # print(k_neighbours)

    while max(k_neighbours) == min(neighbours):
        smallest = min(neighbours)
        k_neighbours.append(smallest)
        neighbours[neighbours.index(smallest)] = math.inf
        if neighbours == [math.inf] * len(neighbours):
            break

    # print(k_neighbours)

    combine_values = []
    for i in range(len(k_neighbours)):
        for j in range(len(og)):
            # print(k_neighbours[i], og[j])
            if k_neighbours[i] == og[j]:
                combine_values.append(examples_copy[j][1])
                og[j] = math.inf
                break
    
    # print(combine_values)

    return combine(combine_values)