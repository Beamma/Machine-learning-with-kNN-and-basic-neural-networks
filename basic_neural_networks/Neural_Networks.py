def construct_perceptron(weights, bias):
    """Returns a perceptron function using the given paramers."""
    def perceptron(input):
        total = 0
        for i in range(len(weights)):
            total += weights[i]*input[i]
        
        total += bias

        if total >= 0:
            return 1

        return 0
    
    return perceptron # this line is fine


def accuracy(classifier, inputs, expected_outputs):
    total = 0
    for i in range(len(inputs)):
        if classifier(inputs[i]) == expected_outputs[i]:
            total += 1
    
    return total / len(inputs)


def learn_perceptron_parameters(weights, bias, training_examples, learning_rate, max_epochs):


    for i in range(max_epochs):

        for example in training_examples:
            perceptron = construct_perceptron(weights, bias)
            y = perceptron(example[0])
            # print("WEIGHTS BIAS PERCEPTION", weights, bias, y)
            
            t = example[1]
            
            x = example[0]
            for i in range(len(x)):
                weights[i] += learning_rate * x[i] * (t-y)
            
            bias += learning_rate * (t-y)

            
    return(weights, bias)

