# Binary Coded Genetic Algorithm
import random

# TODO:
# can a global solution be found?

class Bcga:
    def __init__(self, key, loot, agents, crossover, mutation):
        self.key = key
        self.loot = loot
        self.agents = agents # try and have an even amount of agents
        self.winners = []
        self.kids = []
        self.crossover_rate = round(len(agents[0]) * crossover) 
        self.mutation_rate = mutation # rate for mutation

    # returns the fitness for all agents
    # just for visualization & stopping
    def overall_fitness(self):
        summation = 0
        for agent in self.agents:
            summation += self.fitness(agent)
        return summation 

    # fitness - determines the value on a chromosome
    def fitness(self, chromosome):
        # create fitness value
        fitness = 0

        # create error table
        error = []
        for i in range(0, len(chromosome)):
            error.append(0)

        # loop throguh all items in the loot table, adding their
        # value if the chromosome bit is high
        for itemID in range(0, len(self.loot)):
            fitness += self.loot[itemID][0] * chromosome[itemID]
             
            for valueID in range(1, len(self.loot[itemID])): # loop through non-zero item attributes
                error[valueID - 1] += round(self.loot[itemID][valueID] * chromosome[itemID], 2)
               
                # check if the key limit has been exceeded 
                if (error[valueID - 1] > key[valueID][1]):
                    return 0

        # return the fitness score
        return fitness

    # roulette selection - determines which chromosomes are passed down
    def roulette_selection(self):
        # init values
        participants = self.agents[:]
        self.winners = []
       
        wheel = []
        # create the roulette wheel
        for ID in range(len(participants)):
            fitness = self.fitness(participants[ID])
            for i in range(0, fitness):
                wheel.append(ID)

        # runs roulette twice
        for i in range(0, 2):
            # obtain radom agent from "spinning the roulette wheel"
            winner_ID = wheel[random.randint(0, len(wheel) - 1)]

            # add the agent to the list of winners
            self.winners.append(participants[winner_ID])

    # tournament selection - determines which chromosomes are passed down
    # doesn't work well with a high number of attributes
    def tournament_selection(self):
        # init values
        participants = self.agents[:]
        self.winners = []
        
        # tournament selection
        for i in range(0, 2):
            # obtain random agents and their fitness
            agent_one = participants.pop(random.randint(0, len(participants) - 1))
            agent_one_fitness = self.fitness(agent_one)

            agent_two = participants.pop(random.randint(0, len(participants) - 1))
            agent_two_fitness = self.fitness(agent_two)
            
            # compare the fitness and add winner to winners list
            if (agent_one_fitness > agent_two_fitness):
                self.winners.append(agent_one)
            else:
                self.winners.append(agent_two)

    # crossover - will perform a crossover between parents and create offspring
    def crossover(self):
        # init values
        parents = [self.winners[-1], self.winners[-2]]
        agents_buffer = []

        while len(parents) > 1:
            # get the parents
            mom = parents.pop(0)
            dad = parents.pop(0)

            # create offspring
            daughter = mom[:self.crossover_rate]
            daughter.extend(dad[self.crossover_rate:])
            
            son = dad[:self.crossover_rate]
            son.extend(mom[self.crossover_rate:])
            
            # add offspring as new agents
            agents_buffer.append(daughter)
            agents_buffer.append(son)

        self.kids.extend(agents_buffer)

    # mutation - will randomly flip bits based on the mutation rate
    def mutation(self):
        for agentID in range(len(self.agents)):
            for bitID in range(len(self.agents[agentID])):
                if (random.uniform(0, 1) <= self.mutation_rate):
                    self.agents[agentID][bitID] = (self.agents[agentID][bitID] + 1) % 2

# creates the loot table (blocks) and a key
# total (number of items) & attributes (number of attributes)
def create_inputs (total, attributes):
    # value, weight, volume
    key = [[0, -1]] # -1 means maximize

    # populate key
    for n in range(attributes):
        identifier = random.randint(1, 5) # type of optimization
        key.append([identifier, round(identifier * random.uniform(2, 4), 2)])

    # use key to generate #(total) items with random attributes
    loot = []
    for j in range(total):
        item = []
        for i in range(attributes + 1):
            if (key[i][0] == 0):
                item.append(random.randint(1, 15))
            else:
                item.append(round(random.uniform(1, key[i][0]), 2))
        loot.append(item)

    # return the key & loot table
    return key, loot

# creates an array of agents with binary chromosomes (randomly instantiated)
# total (number of agents) & items (number of items)
def create_bcga_agents(total, items):
    # create list of agents
    agent_list = []

    # populate list of agents
    for i in range(total):
        chromosome = []
        for j in range(items):
            chromosome.append(1 if (i / total) <= random.random() else 0)
        agent_list.append(chromosome)

    # return list
    return agent_list

if __name__ == "__main__":
    # generates the defaults for knapsack simulation
    items = 10                                              # number of items
    attributes = 4                                          # number of attributes
    agents_num = 100                                        # number of agents
    key, loot = create_inputs(items, attributes)            # key table and loot table
    agents = create_bcga_agents(agents_num, items)          # agents/chromosomes
    crossover = .50                                         # crossover rate
    mutation = .01                                          # mutation rate
    history_size = 10                                       # size of fitness record

    # create bcga object
    knapsack = Bcga(key, loot, agents, crossover, mutation)
  
    print("key", knapsack.key)
    print("loot", knapsack.loot)
    print("\n\n")

    fitness_history = [knapsack.fitness(knapsack.agents[0]) - i for i in range(history_size)]

    epoch = 0
    while(1):
        # JUST COOL OUTPUT
        print("epoch", epoch, "\toverall fitness:", knapsack.overall_fitness())
        epoch += 1

        # find best agent
        best_fitness = 0
        best_ID = 0
        for ID in range(len(knapsack.agents)):
            current_fitness = knapsack.fitness(knapsack.agents[ID])
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_ID = ID
        print("best agent:", knapsack.agents[best_ID], "\tfitness:", knapsack.fitness(knapsack.agents[best_ID]))

        fitness_history.pop(0)
        fitness_history.append(best_fitness)

        # ACTUAL LOGIC
        while len(knapsack.kids) < len(knapsack.agents):
            #knapsack.tournament_selection()
            knapsack.roulette_selection()
            knapsack.crossover()
#            print("kids:", len(knapsack.kids))
#            print("agents:", len(knapsack.agents))

        knapsack.agents = knapsack.kids # generation shift
        knapsack.kids = []

        # random mutation
        knapsack.mutation()

        # stop if past 10 results are the same
        continue_training = 0
        for i in range(history_size - 1):
            if (fitness_history[-1 * i] != fitness_history[-1 * (i + 1)]):
                continue_training = 1
                break

        if continue_training == 0:
            break

    print("key", knapsack.key)
    print("loot", knapsack.loot)
    
    
