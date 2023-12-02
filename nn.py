import math
import numpy as np

class NeuralNet(): # acts as a wrapper class for agents and neurons specifically RCGA compatable
    def __init__(self, input_amount, hidden_layers, output_amount, agents_amount=1, activation_type="relu", initialization_type="uniform", alpha=.001, data=None, targets=None):
        self.key = [input_amount] + hidden_layers + [output_amount]             # list referencing # of neurons per layer
        self.agents_amount = agents_amount                                      # number of agents
        self.activation_type = activation_type                                  # activation type of neurons
        self.initialization_type = initialization_type                          # method to use for initializing weights
        self.alpha = alpha                                                      # leaky relu alpha for vals < 0
        self.nuerons = self.create_network()                                    # matrix of neurons
        self.weights = self.create_weights()                                    # Creates variable number of agents
        self.data = data                                                        # dataset
        self.targets = targets                                                  # targets


    # creates a matrix of neurons expressed by the hyperparameters
    def create_network(self):
        return [ [ Neuron(self.activation_type, alpha=self.alpha) for j in range(amount) ] for amount in self.key ] 

    # creates a vector of agents
    def create_weights(self):
        return [Agent(self.key[0], self.key[1:-1], self.key[-1], initialization_type=self.initialization_type, standard_deviation_value=.1) for i in range(self.agents_amount)]

    def mean_squared_error(self, pred, actual):
        return math.pow((actual - pred), 2)

    def gamblers_fitness(self, pred, actual):
        alpha = pred / actual
        if alpha == 1: 
            return 1500
        alpha = abs(alpha - 1)
        if (alpha < 1) and (alpha > 0):
            return abs(1 - alpha) * 1000
        m = math.log10(alpha)
        return abs(50 - m)

    def fitness(self, agent, data=[], targets=[]): # for cali dataset
        if len(data) == 0 and len(targets) == 0:
            data = self.data
            targets = self.targets
        fitness = 0
        for input_vector, expected_output_vector in zip(data, targets):
            predicted = self.evaluate(input_vector, agent)[0]
            fitness += self.gamblers_fitness(predicted, expected_output_vector)
        return fitness

    # returns a vector of outputs (given an input vector, and agent) the size of the number of ouputs
    def evaluate(self, inputs, agent):
        vector = inputs
        for layer in agent.weights:
            vector = self.compute_layer(vector, layer)
        return vector

    # returns vector that is the input for the next layer (or is the output)
    def compute_layer(self, inputs, layer_weights):
        return [ self.compute_neuron(inputs, weights) for weights in layer_weights ]

    # sadge my data structure is unoptimized :( 
    # Lukefahr would be sad im doin this in python
    def compute_neuron(self, inputs, layer_weights):
        summation = 0
        for j in range(len(inputs)):
            for i in range(len(layer_weights)):
                summation += inputs[j] * layer_weights[i]
        return summation

    # will recalculate the fitness value for all agents and update their repsective fitness scores
    def update_fitness(self):
        for agent in self.weights:
            agent.fitness = self.fitness(agent)

    # returns the best (elitism rate * agents_amount) number of agents back
    def elitism(self, elitism_rate):
        num_of_agents = int(elitism_rate * self.agents_amount)
        best_of_best = [self.weights[num_of_agents - i] for i in range(num_of_agents)]
        for agent in self.weights:
            for ID in range(num_of_agents):
                if agent.fitness > best_of_best[ID].fitness:
                    best_of_best[ID] = agent
                    break
        return best_of_best
                    

class Neuron():
    def __init__(self, activation_type, default_value=0, alpha=.001):
        self.activation_type = activation_type
        self.value = default_value
        self.post_value = self.activation()
        self.alpha = alpha # alpha is our leaky relu slope for values less than 0

    def sigmoid_activation(self): # you have to normalize inputs
        return 1 / (1 + math.pow(math.e, (-1 * self.value)))

    def relu_activation(self):
        return max(0, self.value)

    def leaky_relu_activation(self):
        return max(self.alpha * self.value, self.value)

    def parametric_relu_activation(self):
        # similar to leaky relu however we must also have alpha change
        # this change is done through backpropagation or another training
        # method (RCGA)
        #
        # you implement this in my current codebase you should add self.alpha
        # to the values that are being used in the agents selection, crossover,
        # and mutation events.
        return max(self.alpha * self.value, self.value)

    def activation(self):
        if self.activation_type == "sigmoid":
            return self.sigmoid_activation()
        elif self.activation == "relu":
            return self.relu_activation()
        elif self.activation == "lekay relu":
            return self.leaky_relu_activation()
        elif self.activation == "parametric relu":
            return self.parametric_relu_activation()

class Agent():
    def __init__(self, input_amount, layers_key, output_amount, initialization_type="uniform", standard_deviation_value=.1):
        self.key = [input_amount] + layers_key + [output_amount]                # number of neurons in each layer
        self.initialization_type = initialization_type                          # string of initialization type
        self.standard_deviation_value = standard_deviation_value                # standard dev. value for weight init
        self.weights = self.initialize()                                        # weight matrix
        self.fitness = 1 

    def initialize(self): # creates an array, an element for each layer, which consists of arrays for each neuron.
        return [ [ [ self.initial_weight() for k in range(self.key[i]) ] for j in range(self.key[i]) ] for i in range(len(self.key)) ]

    def random_initial_weight(self):
        return np.random.normal(loc=0, scale=self.standard_deviation_value)

    def xavier_glorot_initial_weight(self):
        return np.random.normal(loc=0, scale=(1 / self.key[0]))

    def he_initial_weight(self):
        return np.random.normal(loc=0, scale=(2 / self.key[0]))

    def uniform_initial_weight(self):
        return np.random.uniform(0,1)

    def initial_weight(self):
        if self.initialization_type == "zeros":
            return 0
        elif self.initialization_type == "random":
            return self.random_initial_weight()
        elif self.initialization_type == "xavier glorot":
            return self.xavier_glorot_initial_weight()
        elif self.initialization_type == "he":
            return self.he_initial_weight()
        elif self.initialization_type == "uniform":
            return self.uniform_initial_weight()

    def reproduce(self, weights):
        kid = Agent(self.key[0], self.key[1:-1], self.key[-1], self.initialization_type, self.standard_deviation_value)
        kid.weights = weights
        return kid

    def display_weight_table(self):
        for layer in self.weights:
            print(" | ", end="")
            for neuron in layer:
                print(neuron, end=" |")
            print()

