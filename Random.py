#
# file = open("newfile.txt", "w")
#
#
# LT = ['< 1 week', '1 - 6 weeks', '> 6 weeks']
# Variability = ['L', 'M', 'H']
# Volume = ['< 2.5K pcs/year', '2.5K - 10K pcs/year', '> 10K pcs/year']
# assemblyContent = ['< 1.5%', '1.5% - 40%', '> 40%']
#
# for i in LT:
#     for j in Variability:
#         for k in Volume:
#             for l in assemblyContent:
#                 s = i + "\t" + j+ "\t" + k +"\t" + l
#                 file.writelines(s + "\n")
#
# file.close()

import math
import numpy as np
import scipy.optimize
from pyomo.environ import *
from pyomo.opt import SolverFactory #Import interfaces to solvers
#import pyomo.opt
import ad

print()

#Problem 4
def prob(x):
    return [(x[0]-1)+x[1],x[0]+(x[1]-1)]


def weighting_method_pyomo(f,w):
    points = []
    for wi in w:
        model = ConcreteModel()
        model.x = Var([0,1])
        #weighted sum
        model.obj = Objective(expr = wi[0]*f(model.x)[0]+wi[1]*f(model.x)[1])
        opt = pyomo.opt.SolverFactory("ipopt") #Use ipopt
        #Combination of expression and function
        res=opt.solve(model) #Solve the problem
        points.append([model.x[0].value,model.x[1].value]) #We should check for optimality...
    return points

w = np.random.random((500,2)) #500 random weights
repr = weighting_method_pyomo(prob,w)

f_repr_ws = [prob(repri) for repri in repr]
fig = plt.figure()
plt.scatter([z[0] for z in f_repr_ws],[z[1] for z in f_repr_ws])
plt.show()

fig = plt.figure()
plt.scatter([x[0] for x in repr],[x[1] for x in repr])
plt.show()