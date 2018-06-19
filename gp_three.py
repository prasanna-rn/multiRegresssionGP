from deap import creator, base 
from deap import tools 
from deap import algorithms
from deap import gp

import operator 
import numpy as np 
import math 
import random 
import regression_multi as rm 
import logging, sys

#x = [np.linspace (90,130,500), np.linspace(20,40,500), np.linspace(2,4,500) ]
x = np.matrix ([np.linspace (90,130,500), np.linspace(20,40,500), np.linspace(2,4,500) ])
print (x.shape)
print (x)
#print (x)
def safeDiv (left, right): 
    if right == 0: 
        return 0
    else : 
        return left/ right 


y =  np.matrix ([ -0.08671535, 0.02004189, -0.00111208 ]) * x
y = np.squeeze (np.asarray(y))
print (y.shape)
#y += np.random.normal (size=len(x))
coeffs = np.matrix ( rm.regression (x, y) ) 
#coeffs = coeffs [0]
#print (coeffs)

pset = gp.PrimitiveSet ("MAIN", arity=3)
pset.addPrimitive (np.add, 2)
pset.addPrimitive (np.subtract, 2)
pset.addPrimitive (np.multiply, 2)

range = [-30,30]
def scale (arg):
    return arg * random.randint (range[0], range[1])

#pset.addPrimitive (safeDiv, 2) 
pset.addPrimitive (math.sin, 1)
pset.addPrimitive (math.cos, 1)
pset.addPrimitive (math.tan, 1)
pset.addPrimitive (math.sinh, 1)
pset.addPrimitive (math.cosh, 1)
#pset.addPrimitive (math.exp, 1)
pset.addPrimitive (scale, 1)

pset.renameArguments (ARG0='x')
pset.renameArguments (ARG1='y')
pset.renameArguments (ARG2='z')

creator.create ("FitnessMin", base.Fitness, weights=(-1,))
creator.create ("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def reg_model (xdata, op):

    return rm.model (xdata, 0.5,0.5,0.5)

def evalSymReg (individual): 
    func = toolbox.compile (expr=individual)
    values_one   = x [:,0]
    values_two   = x [:,1]
    values_three = x [:,2]
    
    diff_func = lambda x,y,z: ( func (x,y,z) -  reg_model ( [x,y,z], coeffs ) ) ** 2
    fitness = math.sqrt( sum( map (diff_func, values_one, values_two, values_three) ) )
    return fitness, 

toolbox.register("evaluate", evalSymReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
    randnum_seed = random.randint(1, 500)
    random.seed(randnum_seed)
    
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)
    
    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 50, stats,hof)
    logging.info("Best individual is %s, %s", toolbox.evaluate(hof[0]), hof[0].fitness)
    print (hof[0])
    
    return pop, stats, hof

if __name__ == "__main__":
    main()

