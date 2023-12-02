# Genetic Algorithm Project
This project was for the "Advanced Undergraduate Engineering Math Methods" class (MATH-E449). We were tasked to find and explain an algorithm that we found interesting. I have been interested in genetic algorithms for a while. I coded two types of genetic algorithms: binary coded and real coded.

The general flow of logic for all genetic algorithms are as follows:
1. Identify a fitness function to evaluate your agents/genes
2. Select agents/genes for reproduction based on their fitness score
3. Crossover the agents/genes that were selected
4. Mutate the agents/genes of the next generation
5. Go back to step 2 unless you have reached convergence or a max generation limit

## Selection Methods
This section will highlight different types of selection methods, the idea of each are capable of being implemented for both binary coded and real coded genetic algorithms.

### Tournament Selection
Tournament selection is a method for choosing parents where we pick two random agents from a generation and whichever of the two has a higher fitness score will be a parent. We can run this type of selection twice and will then have two parents that we can use for crossover.

### Roulette Wheel Selection
Roulette wheel selection is a method of choosing parents were a "wheel" is created where each agent in a generation is given a slice of the wheel proportional to the fitness that they have.

![Roulette wheel selection example | Download Scientific Diagram](https://www.researchgate.net/publication/251238305/figure/fig2/AS:335552218976261@1457013291551/Roulette-wheel-selection-example.png)
Once this is done we "spin" the wheel and whatever we land on is the agent that will be chosen for crossover.

The way I implemented this code is by IDing each agent, then creating a list of these IDs where the number of entries for each ID was the corresponding agent's fitness value.

## Crossover
This section goes over different crossover methods. These **are** methods that are specific to binary coded and real coded genetic algorithms; this means a crossover function for a bcga will not work if used on a rcga this is because the genes for these algorithms are inherently different.

### Blend Crossover (Binary Coded) 
Binary crossovers consist of different ways of splicing a binary string and gluing them back together with a portion of another binary string.

![10 Examples of multi-point crossover methods applied to a binary... |  Download Scientific Diagram](https://www.researchgate.net/publication/265505143/figure/fig40/AS:669076007432196@1536531559600/Examples-of-multi-point-crossover-methods-applied-to-a-binary-encoded-solution-In-a.png) Example (a) shows a 50% split between two parents. Both children are made up of half the gene of the parents. Example (b) shows an example where each bit slot in the parent has a 50% (or other variable chance) to show up in the children. This means, for offspring 1, for the first bit that makes up the string, there is a 50% chance it came from chromosome 1 and a 50% chance it came from chromosome 2. Offspring 2 will be the complement to the bit inheritance of offspring 1.

### Linear Crossover (Real Coded)
To compute a linear crossover assume two parents P1 and P2 (real values) create three children $$C_1 = 0.5(P_1 + P_2) \newline C_2 = 1.5P_1 - 0.5P_2 \newline C_3 =1.5P_2 - 0.5 P_1$$

After doing this, find the fitness value for each child, the two children with better fitness will be allowed to be apart of the next generation. The third is removed.

### Blend Crossover (Real Coded)

For real coded genetic algorithms a blend crossover take the following form described in this section.
1. Generate a random number `r` between 0 and 1.
2. Compute gamma $$\gamma = (1 + 2  \alpha)r - \alpha$$ where alpha is a hyper-parameter.
3. Then our two children are described as $$C_1 = (1 - \gamma)P_1 + \gamma P_2 \newline C_2 = \gamma P_1 + (1 - \gamma ) P_2$$

### Simulated Binary Crossover (SBX) (Real Coded)
Simulated binary crossover takes a statistical approach to defining our children. I will work backwards and define the needed variables and concepts as they arise. We consider a hyper-parameter `q` for this crossover, this value will determine how varied the children are.

 1. We consider two children as defined by $$C_1 = 0.5(P_1 + P_2) - \alpha' * |P_2 - P_1| \newline C_2 = 0.5(P_1 + P_2) + \alpha' * |P_2 - P_1|$$
 2. "alpha prime" is a value that we compute through one of three methods depending on what type of "crossover event" we choose. To choose it, we pick a random number `r` between 0 and 1. Then if `r > 0.5` we consider an expanding crossover event, if `r = 0.5` then we consider a stationary crossover event, if `r < 0.5` we consider a contracting crossover event.

Each crossover event is based on the following polynomial probability distribution (`q` is what changes the graph in the below gif):

![sbxgif](https://i.ibb.co/0FkGMV2/sbxgif.gif)

#### Expanding Crossover Event
This is an example of the result for a expanding crossover event.
![expanding-crossover](https://i.ibb.co/4F10DDJ/expanding-crossover.jpg)
$$\alpha' = 2r^{q + 1}$$
This is derived from the area under the curve of the right side.
$$\int_{\alpha'}^{\infty }0.5(q+1)\frac{1}{\alpha^{q+2}} = r $$
$$0.5(q+1)\int_{\alpha'}^{\infty }\alpha^{-q-2} = r $$
$$\frac{-0.5}{\infty^{q+1}} - \frac{-0.5}{\alpha'^{q+1}} = r $$
$$0.5\alpha'^{-q-1} = r $$
$$\alpha'^{-q-1} = 2r \newline$$
$$\alpha' = 2r^{q+1}$$

#### Stationary Crossover Event
This is an example of the result for a stationary crossover event.
![stationary-crossover](https://i.ibb.co/WPjm6gW/stationary-crossover.jpg)
$$\alpha' = 1$$
This exists at the center line; hence alpha prime is one.
#### Contracting Crossover Event
This is an example of the result for a contracting crossover event.
![contracting-crossover](https://i.ibb.co/xLLhSbj/contracting-crossover.jpg)
$$\alpha' = 2r^{\frac{1}{q + 1}}$$
This is derived from the area under the curve of the left side.
$$\int_{0}^{\alpha' }0.5(q+1)\alpha^{q} = r $$
$$0.5(q+1)\int_{0}^{\alpha' }\alpha^{q} = r $$
$$0.5\alpha'^{q+1}-0.5(0)^{q+1} = r $$
$$\alpha'^{q+1} = 2r $$
$$\alpha' = 2r^{\frac{1}{q+1}}$$

## Mutation
Mutation events are similar to our crossover functions in the sense that binary coded and real coded genetic algorithms implement different mutation functions. I will specifically go over the three mutation functions implemented in my code.

### Random Mutation (Binary Coded)
Binary coded mutation have a hyper-parameter relating to the rate at which each bit has to flip.

### Random Mutation (Real Coded)
Real coded genetic algorithms have a similar function to that of the binary coded genetic algorithm's random mutation function. In this case we will consider a hyper-parameter, $\delta$, which will refer to the variance each mutation can take. We also consider a random value between 0 and 1, $r$. $P$ is our gene undergoing mutation.$$(P + r -0.5)  \delta$$

### Polynomial Mutation (Real Coded)
Polynomial mutation for real coded genetic algorithms will take a random number between 0 and 1, $r$, a hyper-parameter $\delta$, to determine variance of each mutation, and calculate a perturbation factor, $\bar{\delta}$, which will be used to mutate a gene.
$$\bar{\delta} =
    \begin{cases}
      (2r)^{\frac{1}{q+1}} & \text{, if r < 0.5}\\
      1-(2(1-r))^\frac{1}{q+1} & \text{, if r ≥ 0.5}\\
    \end{cases}       $$
    From this we define the mutation as $$P +\bar{\delta}\delta$$

## Binary Coded Genetic Algorithm
The `bcga.py` file is an example of the knapsack problem, a toy problem where we are trying to maximize some value while remaining in within the constraints of the input's attributes. 

Imagine we enter a gene's cave full of riches in the middle of the desert at night. We want to leave with our bag full of the riches, but we can only carry a certain amount of weight. In this case we can apply a binary coded genetic algorithm.

Assume a "gene" as a series of binary values that represent whether or not a certain input is included in our bag. We create a "generation" of these genes or "agents" (as I will call them in our rcga implementation) from many of these genes. We can evaluate how good each gene preforms through a "fitness function" which will, in our case, provide the total value of each item that was included in our solution. We go through a selection process to determine which genes are fit to reproduce and then a crossover event happens between the two parents. The resultant children are a mix of the bits that made up the two parents. The last step before moving to our next generation is a brief mutation step. We will randomly flip some of the bits that make up each gene to maintain some genetic diversity. We repeat this process of selection, crossover, and mutation until we find some convergence or until a certain generation number has been hit.

## Real Coded Genetic Algorithm
This portion of the project is split up over three files: `rcga.py` `nn.py` and `demo_rcga.py`. I developed a neural net from the ground up for transparency on a real coded genetic algorithm being a process for training one. In the demo we load in the California Housing Dataset from scikitlearn and use a neural net to predict the median house value for California districts.

For the example case in this repository consider the weights and biases of a neural net as the "genes" or "agents" of a "generation". We create many different agents (weights and biases) for a neural net and will evaluate the performance of each agent with a custom fitness function.

    def gamblers_fitness(self, pred, actual):
        alpha = pred / actual
        if alpha == 1: 
            return 1500
        alpha = abs(alpha - 1)
        if (alpha < 1) and (alpha > 0):
            return abs(1 - alpha) * 1000
        m = math.log10(alpha)
        return abs(50 - m)
I developed this fitness function with roulette wheel selection in mind. I found when I used an error function (like mean squared error) my roulette wheel selection function did not provide reliable outputs. This `gamblers_fitness` function will reward predicted values closer to the actual value.

We take the same logical flow of most genetic algorithms (described in the first section). We will take our fitness function, evaluate our agents with this function; then select the agents for reproduction using a selection method (roulette wheel selection). Something to note is that I also implemented elitism to make sure the best agents of a generation would be carried on to the next generation with no difference. This elitism is to ensure that the best agents will not be genetically muddled with worse genes. After selection we undergo crossover similar to the binary coded genetic algorithm however we are now working with real values. Because of this crossover is a bit different. I implemented a few crossover functions but decided to use Simulated Binary Crossover (SBX) for the demo. After crossover we go through our gene and mutate the values a small amount based on our mutation rate hyper-parameter. Because convergence is much more difficult to find in our case I ran the genetic algorithm over a limited number of generations.
