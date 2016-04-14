import math
import scipy.stats as st
from scipy.optimize import minimize
import ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_data():
    df = pd.read_csv('Data-TIES583.csv',encoding='latin-1')








def inventory_prob(x,y):
    setup_cost = 10.
    est_demand = 15.
    holding_cost = 0.5
    price = 20.
    cost = 6.
    LT = 1.2
    std_est_demand = 12.5
    ss = st.norm.ppf(0.95)*math.sqrt((LT*(std_est_demand**2)+est_demand))
    return [est_demand*setup_cost/x+est_demand*cost+(holding_cost*y**2)/(2*x)+ price*(x-y)**2/(2*x),\
           ss *x/est_demand]


def calc_ideal(f):
    ideal = [0]*2 #Because three objectives
    solutions = [] #list for storing the actual solutions, which give the ideal
    bounds = ((1.,20.),(1.,13.)) #Bounds of the problem
    for i in range(2):
        res=minimize(
            #Minimize each objective at the time
            lambda x: f(x[0],x[1])[i], [1,1], method='SLSQP'
            #Jacobian using automatic differentiation
            ,jac=ad.gh(lambda x: f(x[0],x[1])[i])[0]
            #bounds given above
            ,bounds = bounds
            ,options = {'disp':True, 'ftol': 1e-20, 'maxiter': 1000})
        solutions.append(f(res.x[0],res.x[1]))
        ideal[i]=res.fun
    return ideal,solutions

ideal, solutions= calc_ideal(inventory_prob)
print ("ideal is "+str(ideal))


def inventory_prob_normalized(x,y):
    z_ideal = [104.3902330623304, 1.5604451636266721]
    z_nadir = [240.25,21.40854560]
#    import pdb; pdb.set_trace()
    z = inventory_prob(x,y)
    return [(zi-zideali)/(znadiri-zideali) for
            (zi,zideali,znadiri) in zip(z,z_ideal,z_nadir)]


# ####### Weighting Method ##########33
# def weighting_method(f,w):
#     points = []
#     bounds = ((1.,20.),(1.,13.)) #Bounds of the problem
#     for wi in w:
#         res=minimize(
#             #weighted sum
#             lambda x: sum(np.array(wi)*np.array(f(x[0],x[1]))),
#             [1,1], method='SLSQP'
#             #Jacobian using automatic differentiation
#             ,jac=ad.gh(lambda x: sum(np.array(wi)*np.array(f(x[0],x[1]))))[0]
#             #bounds given above
#             ,bounds = bounds,options = {'disp':False})
#         points.append(res.x)
#     return points
#
# w = np.random.random((500,2)) #500 random weights
# repr = weighting_method(inventory_prob_normalized,w)
#
# ###### Plotting #####
# f_repr_ws = [inventory_prob(repri[0],repri[1]) for repri in repr]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter([f[0] for f in f_repr_ws],[f[1] for f in f_repr_ws])
# plt.show()



######## epsilon - constraint method ###########
def e_constraint_method(f,eps,z_ideal,z_nadir):
    points = []
    for epsi in eps:
        bounds = ((1.,epsi[0]*(z_nadir[0]-z_ideal[0])+z_ideal[0]),
                  ((epsi[1]*(z_nadir[1]-z_ideal[1])+z_ideal[1]),
                   40.)) #Added bounds for 2nd objective
        res=minimize(
            #Second objective
            lambda x: f(x[0],x[1])[0],
            [1,1], method='SLSQP'
            #Jacobian using automatic differentiation
            ,jac=ad.gh(lambda x: f(x[0],x[1])[0])[0]
            #bounds given above
            ,bounds = bounds,options = {'disp':False})
        if res.success:
            points.append(res.x)
    return points


z_ideal = [104.3902330623304, 1.5604451636266721]
z_nadir = [240.25,21.40854560]
eps = np.random.random((100,2))
repr = e_constraint_method(inventory_prob_normalized,eps,z_ideal,z_nadir)

f_repr_eps = [inventory_prob(repri[0],repri[1]) for repri in repr]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([f[0] for f in f_repr_eps],[f[1] for f in f_repr_eps])
plt.show()