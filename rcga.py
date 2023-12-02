# Real Coded Genetic Algorithm
# implemented for neural nets
import random
import math

# GOAL: make it generic to be applicable for any operation

class Rcga():
    # init
    def __init__(self, blend_alpha = .5, q = 2, random_mutation_delta = .01, polynomial_mutation_delta_max = .01):
        self.blend_alpha = blend_alpha                                      # blend crossover hyperparameter
        self.random_mutation_delta = random_mutation_delta                  # random mutation hyperparameter
        self.polynomial_mutation_delta_max = polynomial_mutation_delta_max  # polynomial mutaion hyperparameter
        self.q = q                                                          # simulated binary crossover and polynomial mutation hyperparameter

    # Selection (to work with the homemade neural net class I wrote)
    # Pick two parents to reproduce
    def tournament_selection(self, network):
        # get two random agents 
        challengers = random.sample(network.weights, 2)

        # get their fitness
        challenger_one_fitness = network.fitness(challengers[0]) 
        challenger_two_fitness = network.fitness(challengers[1])

        # choose and return better challenger
        if challenger_one_fitness > challenger_two_fitness:
            return challengers[0]
        else:
            return challengers[1]

    def roulette_selection(self, network): # this is an augmented version of typicall roulette wheel
        # create roulette wheel of ID's
        wheel = []
        for ID in range(len(network.weights)):
            if network.weights[ID].fitness > 0:
                slice_count = network.weights[ID].fitness
            else:
                slice_count = 1  # something catastrophic occured
            for amount in range(int(slice_count)):
                wheel.append(ID)

        # return the winners
        return [ network.weights[wheel[random.randint(0, len(wheel) - 1)]], network.weights[wheel[random.randint(0, len(wheel) - 1)]] ]
 
    # Crossover
    # crossover functions will return a list of two children 
    # parents are full weights
    def linear_crossover(self, parent_one, parent_two, network):
        # create children
        child_one = [ [ [ (.5 * (weight_one + weight_two)) for weight_one, weight_two in zip(junction_one, junction_two) ] for junction_one, junction_two in zip(layer_one, layer_two) ] for layer_one, layer_two in zip(parent_one, parent_two) ]
        child_two = [ [ [ (1.5 * weight_one - .5 * weight_two) for weight_one, weight_two in zip(junction_one, junction_two) ] for junction_one, junction_two in zip(layer_one, layer_two) ] for layer_one, layer_two in zip(parent_one, parent_two) ]
        child_three = [ [ [ (1.5 * weight_two - .5 * weight_one) for weight_one, weight_two in zip(junction_one, junction_two) ] for junction_one, junction_two in zip(layer_one, layer_two) ] for layer_one, layer_two in zip(parent_one, parent_two) ]

        # find the fitness of each child
        child_one_fitness = network.fitness(child_one) 
        child_two_fitness = network.fitness(child_two) 
        child_three_fitness = network.fitness(child_three) 

        # return the better two children
        temporary_list = [ (child_one_fitness, child_one), (child_two_fitness, child_two), (child_three_fitness, child_three) ]
        temporary_list.sort(key=lambda x: x[0])
        return [ temporary_list[0], temporary_list[1]]

    def blend_crossover(self, parent_one, parent_two): # BLX-alpha
        # init vars needed for computation
        r = random.random()
        gamma = ((1 + 2 * self.blend_alpha) * r) - self.blend_alpha
        
        # compute children
        child_one = [ [ [ (((1 - gamma) * weight_one) + (gamma * weight_two)) for weight_one, weight_two in zip(junction_one, junction_two) ] for junction_one, junction_two in zip(layer_one, layer_two) ] for layer_one, layer_two in zip(parent_one, parent_two) ]
        child_one = parent_one.reproduce(child_one)
        child_two = [ [ [ (((1 - gamma) * weight_two) + (gamma + weight_one)) for weight_one, weight_two in zip(junction_one, junction_two) ] for junction_one, junction_two in zip(layer_one, layer_two) ] for layer_one, layer_two in zip(parent_one, parent_two) ]
        child_two = parent_two.reproduce(child_two)
       
        # return children
        return [child_one, child_two]

    def simulated_binary_crossover(self, parent_one, parent_two): # SBX
        # init vars needed for computation
        r = random.random()
        alpha_prime = 0
        if r < .5:
            alpha_prime = self.contracting_crossover(r)
        elif r == 1:
            alpha_prime = self.stationary_crossover()
        elif r > .5:
            alpha_prime = self.expanding_crossover(r)

        # compute children
        child_one = [ [ [ (.5 * ((weight_one + weight_two) - alpha_prime * abs(weight_two - weight_one))) for weight_one, weight_two in zip(junction_one, junction_two) ] for junction_one, junction_two in zip(layer_one, layer_two) ] for layer_one, layer_two in zip(parent_one.weights, parent_two.weights) ]
        child_one = parent_one.reproduce(child_one)

        child_two = [ [ [ (.5 * ((weight_one + weight_two) + alpha_prime * abs(weight_two - weight_one))) for weight_one, weight_two in zip(junction_one, junction_two) ] for junction_one, junction_two in zip(layer_one, layer_two) ] for layer_one, layer_two in zip(parent_one.weights, parent_two.weights) ]
        child_two = parent_two.reproduce(child_two)

        # return children
        return [child_one, child_two]

    def contracting_crossover(self, r):
        return math.pow((2 * r), (1 / (self.q + 1)))

    def stationary_crossover(self):
        return 1

    def expanding_crossover(self, r):
        return math.pow((2 * r), (self.q + 1))

    # Mutation
    # mutation functions returns mutated value
    def random_mutation(self, parent):
        weights = [ [ [ (weight + (random.random() - .5) * self.random_mutation_delta) for weight in junction ] for junction in layer ] for layer in parent.weights ]
        return parent.reproduce(weights)

    def polynomial_mutation(self, parent):
        r = random.random()
        delta_bar = 0
        if r < .5:
            delta_bar = math.pow( (2 * r), (1 / (self.q + 1)) )
        else:
            delta_bar = 1 - math.pow( (2 * (1 - r)), (1 / (self.q + 1)) )
        weights = [ [ [ (weight + delta_bar * self.polynomial_mutation_delta_max) for weight in junction ] for junction in layer ] for layer in parent.weights ]
        return parent.reproduce(weights)


