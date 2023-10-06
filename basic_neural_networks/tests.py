from knn import euclidean_distance, majority_element, knn_predict
from Neural_Networks import construct_perceptron, accuracy, learn_perceptron_parameters

def test_1():
    print("TEST 1:")
    print(euclidean_distance([0, 3, 1, -3, 4.5],[-2.1, 1, 8, 1, 1]))
    print(majority_element([0, 0, 0, 0, 0, 1, 1, 1]))
    print(majority_element("ababc") in "ab")
    print("")


def test_2():
    print("TEST 2:")

    examples = [
        ([2], '-'),
        ([3], '-'),
        ([5], '+'),
        ([8], '+'),
        ([9], '+'),
    ]

    distance = euclidean_distance
    combine = majority_element

    for k in range(1, 6, 2):
        print("k =", k)
        print("x", "prediction")
        for x in range(0,10):
            print(x, knn_predict([x], examples, distance, combine, k))
        print()

    examples = [
    ([1], 5),
    ([2], -1),
    ([5], 1),
    ([7], 4),
    ([9], 8),
    ]

    def average(values):
        return sum(values) / len(values)

    distance = euclidean_distance
    combine = average

    for k in range(1, 6, 2):
        print("k =", k)
        print("x", "prediction")
        for x in range(0,10):
            print("{} {:4.2f}".format(x, knn_predict([x], examples, distance, combine, k)))
        print()


def test_3():
    print("TEST 3:")
    weights = [2, -4]
    bias = 0
    perceptron = construct_perceptron(weights, bias)

    print(perceptron([1, 1]))
    print(perceptron([2, 1]))
    print(perceptron([3, 1]))
    print(perceptron([-1, -1]))
    print()

    weights = [1.5, -2.25]
    bias = -3.75
    perceptron = construct_perceptron(weights, bias)

    print(perceptron([1, -4]))
    print(perceptron([0, -5]))
    print(perceptron([-3, 3]))
    print(perceptron([-1, -1]))


def test_4():
    print("TEST 4:")
    perceptron = construct_perceptron([-1, 3], 2)
    inputs = [[1, -1], [2, 1], [3, 1], [-1, -1]]
    targets = [0, 1, 1, 0]

    print(accuracy(perceptron, inputs, targets))

    print()

    
def test_5():
    weights = [2, -4]
    bias = 0
    learning_rate = 0.5
    examples = [
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 0),
    ]
    max_epochs = 50

    weights, bias = learn_perceptron_parameters(weights, bias, examples, learning_rate, max_epochs)
    print(f"Weights: {weights}")
    print(f"Bias: {bias}\n")

    weights = [2, -4]
    bias = 0
    learning_rate = 0.5
    examples = [
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
    ]
    max_epochs = 50

    weights, bias = learn_perceptron_parameters(weights, bias, examples, learning_rate, max_epochs)
    print(f"Weights: {weights}")
    print(f"Bias: {bias}\n")

    perceptron = construct_perceptron(weights, bias)

    print(perceptron((0,0)))
    print(perceptron((0,1)))
    print(perceptron((1,0)))
    print(perceptron((1,1)))
    print(perceptron((2,2)))
    print(perceptron((-3,-3)))
    print(perceptron((3,-1)))


def main():

    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    




main()