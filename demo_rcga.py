from nn import NeuralNet, Neuron, Agent
from rcga import Rcga
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

if __name__ == "__main__":
    # load in and split dataset
    california_housing = fetch_california_housing()
    data, target = california_housing.data, california_housing.target
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.2, random_state=3)

    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.fit_transform(data_test)

    pca = PCA(n_components='mle')
    data_train = pca.fit_transform(data_train)
    data_test = pca.fit_transform(data_test)

    # init neural net 
    input_amount = len(data_train[0])
    hidden_layers = [  4, 3 ]
    output_amount = 1 # beacuse the dataset is predicting the one number
    agents_amount = 40
    activation_type = "relu"

    network = NeuralNet(input_amount, hidden_layers, output_amount, agents_amount, activation_type, data=data_train, targets=target_train)
    
    network.update_fitness() # provides fitness values to all new agents

    # init RCGA training class
    q = 4 
    
    rcga = Rcga(q=q)

    # training process
    kindergarden = []
    print()
    print("==================== GEN wack inputs ====================")
    print()
    generation = 0
    total_gen = 100 
    while generation < total_gen: 
        # Elitism
        if len(kindergarden) == 0:
            print("* Preforming Elitism...")
            kindergarden = kindergarden + network.elitism(.2)
            print("Fitness of Agents Passed Down: [ ", end="")
            for agent in kindergarden:
                print(agent.fitness, end=" ")
            print("]")
            print()

        # selection
        print("* Selecting Parents...")
        parents = rcga.roulette_selection(network)
        print("Parent One Fitness:", parents[0].fitness)
        print("Parent Two Fitness:", parents[1].fitness)
        print()

        # crossover
        print("* Crossing Genes...")
        new_kids = rcga.simulated_binary_crossover(parents[0], parents[1])

        new_kids[0].fitness = network.fitness(new_kids[0])
        new_kids[1].fitness = network.fitness(new_kids[1])

        print("Child One Fitness:", new_kids[0].fitness)
        print("Child Two Fitness:", new_kids[1].fitness)
        print()

        # mutatioa
        print("* Mutating...")
        new_kids[0] = rcga.polynomial_mutation(new_kids[0])
        new_kids[1] = rcga.polynomial_mutation(new_kids[1])

        new_kids[0].fitness = network.fitness(new_kids[0])
        new_kids[1].fitness = network.fitness(new_kids[1])

        print("Post Mutation Child One Fitness:", new_kids[0].fitness)
        print("Post Mutation Child Two Fitness:", new_kids[1].fitness)
        print()

        # add new children to the class
        kindergarden.append(new_kids[0])
        kindergarden.append(new_kids[1])

        print("children pool: ", len(kindergarden), "\t|\t", str(len(kindergarden) / len(network.weights)) + "% close to the next generation")

        generation += 1

        if len(kindergarden) >= len(network.weights):
            
            network.weights = kindergarden
            kindergarden = []
            print("==================== GEN Over ====================")
            print()
        
            if generation != (total_gen):
                print()
                print("==================== GEN", generation, "====================")
                print()
        else:
            generation -= 1

    print("*** Training Over ***")
    # used data_test and target_test to see how good the best agents is:
    best_fitness = 0
    best_agent = 0
    for agent in network.weights:
        if agent.fitness > best_fitness:
            best_fitness = agent.fitness
            best_agent = agent
    print()
    print("* Summary...")
    best_agent.fitness = network.fitness(best_agent, data_test, target_test)
    print("the best agent, named", str(best_agent) + ", had (test) fitness:", best_agent.fitness, "and (train) fitness:", best_fitness)
    print()
    
    print("examples of (train) input/output predictions")
    for i in range(10):
        prediction = network.evaluate(data_train[i], best_agent)
        print("predicted:", round(prediction[0], 3), "\t|\tactual:", target_train[i])

    print()
    print("examples of (test) input/output predictions")
    for i in range(10):
        prediction = network.evaluate(data_test[i], best_agent)
        print("predicted:", round(prediction[0], 3), "\t|\tactual:", target_test[i])


