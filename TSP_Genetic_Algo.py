import random, math, itertools, time
import numpy as np
import matplotlib.pyplot as plt

#In this project, I used genetic algorithms to 
#create a program that finds a near optimal solution
#to a variant of the TSP. In this version, you are given
#a set of coordinates and you have to find the shortest
#path that goes through all of them. The path doesn't 
#have to be a Hamiltonian cycle and the weight assigned to each edge
#is just the Euclidean distance between the two coordinates


#to avoid redundant calculations, we can store the distances between 
#coordinates that we have calculated so far in a dictionary and retrieve those values
#instead of calculating them again (which will definitely happen a lot since
#we're using a GA)
distances = dict()



#here is our fitness function. It computes the total distance
#of a possible path that can be taken

#it takes in the candidate solution, a permutation of the coordinates given
#and the list of the coordinates
def fitness(solution, locations):
  tot_dis = 0
  #for each edge
  for coord in range(len(solution)- 1):
    #generate the corresponding key for that edge which is of the form (x1, y1)(x2, y2)

    key = str(locations[solution[coord]]) + str(locations[solution[coord + 1]])

    #if we haven't calculated the distance between those two points before
    if(not (key in distances.keys())):
      #calculate it and store it in the dictionary, we don't really need to
      #take the square root since the square root is a strictly increasing function
      distances[key] = (locations[solution[coord]][0] - locations[solution[coord + 1]][0])**2 + (locations[solution[coord]][1] - locations[solution[coord + 1]][1])**2
    
    #add the distance to the path length
    tot_dis += distances[key]

  return tot_dis

#here we create an offspring solution from two parent solutions
#I used the Davis' order crossover algorithm which is where 
#you randomly choose two indices, index1 and index2, that represent the beginning and end
#of the subarray of parentA that gets copied into the corresponding "slot" of the child solution.
#You then fill in the rest of the child's array with the entries of parentB
#that aren't already present in the child array starting with the entry in index1
#of parentB.


def crossover(parentA, parentB):
  #allocate space for the offspring solution
  child = [-1]*len(parentA)

  #generate the two random indices
  index1 = random.randint(0, len(parentA))
  index2 = random.randint(0, len(parentA))

  #and make sure that the two indices aren't equal to one another
  while(index2 == index1):
    index2 = random.randint(0, len(parentA))
  temp = index1
  index1 = min(temp, index2)
  index2 = max(temp, index2)

  #Take the subarray of parentA and put it into the corresponding slot
  #of the child array
  child[index1:index2] = parentA[index1:index2]
  
  #after that we fill in the rest of the child array with
  #the contents of parentB

  #we will have to index counters
  #one for the index of the child array
  #and one for the index of parentB which as mentioned before
  #starts from index1
  indexB = index1
  indexChild = 0

  #while we haven't completely filled the child array
  while(indexChild < len(child)):

    #if the entry in the child array has already been filled
    if(child[indexChild] != -1):
      #move on to the next slot in the child array
      indexChild += 1
    
    #else if the value that we're trying to place into the child array
    #is already present in the child array (it came from parentA)
    elif(parentB[indexB % len(parentB)] in parentA[index1:index2]):
      #move onto the next value
      indexB += 1
    
    else:
      #else we fill in the empty slot of child1 with the corresponding
      #entry in parentB. Since indexB doesn't start from 0, we might have
      #to wrap around the parentB array so we mod indexB by the length of parentB
      child[indexChild] = parentB[indexB % len(parentB)]
      #and we also increment both index counters
      indexChild += 1
      indexB += 1
  
  return child

#in order to ensure that we fully explore the solution space
#we need to make sure that our population remains diverse

#this function tweaks the child array by randomly swapping 
#entries in the array

def mutate(child):

  mut_child = child
  #randomly choose two distinct indices
  index1 = random.randint(0, len(child) - 1)
  index2 = random.randint(0, len(child) - 1)
  while(index2 == index1):
    index2 = random.randint(0, len(child) - 1)
  
  #and then swap the corresponding entries
  mut_child[index1], mut_child[index2] = mut_child[index2], mut_child[index1]

  return mut_child

#here we create the two offspring from the two parent arrays
#and then apply a mutation to each child

#takes in the two parent arrays, parentA and parentB
#as well as the mutate_rate which tells us how many times 
#we should mutate the child array
def create_children(parentA, parentB, mutate_rate):

  #create two children from the parentA and parentB
  #but for child2, we swap the roles of parentA and parentB
  #this is done because it's possible that our crossover method
  #might bias the representation of one parent over another in their
  #children's genes. Having the roles swapped helps reduce this potential bias

  child1 = crossover(parentA, parentB)
  child2 = crossover(parentB, parentA)

  #and here we mutate the child arrays however many times we need to
  for iter in range(mutate_rate):
    child1 = mutate(child1)
    child2 = mutate(child2)
  
  return child1, child2

#the solution class bundles together all of the attributes of a 
#candidate solution: the permutation of coordinates it uses, its fitness
#score or its path length, and the "age" of the candidate solution

class solution:
  def __init__(self, permutation, age, score):
    self.permutation = permutation
    self.age = age
    self.score = score
  
#in this function, from the current generation of solutions,
#we create a new generation and then remove the solutions
#that exceed a given age limit. We remove solutions so that
#if we want to use a large number of generations in our algorithm,
#we don't have to worry about a large exponential growth in the population. 
#And we use aged based selection so that we reduce the risk of 
#converging on a local optimum

#we choose which solutions get to pass on their genes through tournament selection:
#randomly choose K solutions and take the most fit to be parent1. Then repeat
#to get parent2

#this function takes in the old generation, a list of solution objects,
#a list of the coordinates, the mutate_rate, and the age limit
def generate(oldGen, locations, mutate_rate, age):
  newGen = oldGen

  #I set the number of children to add to be floor(N/4)*2 where 
  #N is the size of the old generation. So after 3 generations, we will
  #have roughly N + N/2 + 3N/4 individuals in the population. If our age limit is 3, then after that, 
  #we would have a4 = (a1 + a2 + a3)/2, a5 = (a2 + a3 + a4)/2, ... aN = (aN-1 + aN-2 + aN-3)/2.
  #which is a recurrence relation with characteristic equation x^3 - 0.5*x^2 -0.5*x - 0.5 = 0.
  #If the roots are x1, x2, x3 then the explicit formula for the population size at generation k is
  #G(k) = b1*x1^k + b2*x2^k + b3*x3^k.  We can use the initial conditions given by a1, a2, and a3
  #to solve for b1, b2, and b3. Then by the triangle inequality, |G(k)| <= |b1|*|x1|^k + |b2|*|x2|^k
  #+ |b3||x3|^k = N*(0.18742*1.734^k + 0.598*0.6369^k + 0.598*0.6369^k). So with these parameters, 
  #the population will still blow up but the growth isn't as severe as what we would have if
  #we hadn't placed an age limit. 

  numAdd = math.floor(len(oldGen) / 4)

  #to make a new addition
  for count in range(numAdd):
    #we take a random sample of the old generation
    #Here I made the sample size 5 but this is another parameter that you could
    #play around with. A small sample size means that you're more likely to pass 
    #on the genes of individuals who aren't that fit which increases the diversity
    #of the population. But there's also the risk of the most fit not being represented
    #as much as they should be in the new generation

    selection  = random.sample(oldGen, 5)
    #we then take the most fit solution in that sample
    selection.sort(key = lambda x: x.score)

    #and it becomes a parent and gets to pass on its genes
    parentA = selection[0]
    #and we do the same thing to get parentB
    selection  = random.sample(oldGen, 5)
    selection.sort(key = lambda x: x.score)
    parentB = selection[0]

    #we then generate the children
    child1, child2 = create_children(parentA.permutation, parentB.permutation, mutate_rate)

    #and add them to the next generation giving them an age of -1
    newGen.append(solution(child1, -1, fitness(child1, locations)))
    newGen.append(solution(child2, -1, fitness(child2, locations)))
  
  #we then perform the age based selection on the new generation

  #sort the solutions by their age in ascending order
  newGen.sort(key = lambda x: x.age)


  index = 0
  #find the index where we the solutions start to exceed the age limit
  while(index < len(newGen) and newGen[index] != age):
    index += 1
  
  #we then just remove all of the solutions that exceed the age limit
  #from the new generation
  newGen = newGen[0:index - 1]

  #while(index < len(newGen)):
    #newGen.pop()
  
  #finally we increment all of the ages of the solutions
  for individual in newGen:
    individual.age += 1
  
  return newGen
    
  


#here is the main driver function that runs the GA
#it takes in the list of coordinates, the number of generations
#to run the algorithm, the mutation rate, and the age limit
#returns the highest performing solution in the latest generation

def simulate(locations, numGen, mutationRate, age):

  population = []
  #here we generate the initial population which consists
  #of N random permutations of the list of coordinates
  for perm in range(len(locations)):
    individual = np.random.permutation(len(locations))
    population.append(solution(individual, 0, fitness(individual, locations)))

  #then for each iteration
  for iter in range(numGen):
    #we get the next generation from the old one, generally allowing
    #high performing solutions to breed with other fit solutions
    population = generate(population, locations, mutationRate, age)
  
  #once we're done, we get the highest performing solution in the current generation
  
  best = min(population, key = lambda x: x.score)
  return best.permutation, best.score



#here we test the performance of our algorithm
#this function takes in N, the number of locations we need to visit
#and tells us the optimal found as well as the time it took to find that solution

def perf_test(N, verbose):
  
  locations = []
  #we create the coordinates by randomly choosing N lattice points in an NxN box

  for i in range(N):
    locations.append((random.randint(0, N), random.randint(0, N)))

  #then we call on the simulate function to get the optimal solution
  start = time.time()
  #I had it run for 15 generations, set the mutate rate to 2, and the age limit
  #to 3
  result = simulate(locations, 15, 2, 3)
  end = time.time()

  #After that, we print out the best solution and the time it took to find it

  if(verbose):
    print("The optimal path found by the genetic algorithm was: " + str(result[0]))
    print("which has a path length of: " + str(result[1]))

    print("The algorithm found this solution in " +str(end-start) + " seconds.")
  
  return end-start


#in this function we compare the performance of the GA against
#a brute force approach
def benchmark(N):
  locations = []

  for i in range(N):
    locations.append((random.randint(0, N), random.randint(0, N)))
  
  distances = dict()

  #generate all possible permutations of the list of coordinates
  #and then find the shortest path and record the time it took to find the solution
  startBF = time.time()
  perm = list(itertools.permutations(range(len(locations))))
  shortest_path = min(perm, key = lambda x: fitness(x, locations))
  endBF = time.time()


  startGA = time.time()
  result = simulate(locations, 15, 2, 3)
  endGA = time.time()

  #display the results found for both the genetic algorithm as well as the brute
  #force approach
  print("The optimal path found by the genetic algorithm was: " + str(result[0]))
  print("which has a path length of: " + str(result[1]))

  print("The algorithm found this solution in " +str(endGA - startGA) + " seconds.\n\n")


  print("The optimal path found through a brute force approach was: " + str(shortest_path))
  print("which has a path length of: " + str (fitness(shortest_path, locations)))

  print("The algorithm found this solution in " +str(endBF - startBF) + " seconds.")



#finally in order to look at the run time of the algorithm
#I plotted its execution time as a function of the number of 
#coordinates. It looks like it's roughly quadratic which is a significant
#improvement from the brute force approach

def runtime(N):
  execTime = []

  for size in range(5, N):
    execTime.append(perf_test(size, False))
  print(execTime)
  plt.plot(range(5, N), execTime)
  plt.xlabel("Number of Stops")
  plt.ylabel("Execution Time (seconds)")
  plt.show()


runtime(50)