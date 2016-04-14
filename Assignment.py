import math
import numpy as np
import scipy.optimize
#from pyomo.environ import *
#from pyomo.opt import SolverFactory #Import interfaces to solvers
#import pyomo.opt
import ad


import matplotlib.pyplot as plt

#Problem 1
# Objective function will be:
# min x*y + 0.5*pi*r^2
# s.t. 2x + y + pi*r <= 12
# x,y,r >= 0
# where x and y are the sides of rectangle and r is the radius of the semi circle.
def f_objective(x):
    return -1*(x[0] * x[1] + math.pi/2*(x[2]**2))

def f_obj(x):
    return -1*(x[0] * x[1] + math.pi/2*(x[2]**2)), [], [2*x[0]+x[1]+math.pi*x[2],x[0], x[1], x[2]]

def func_deriv(x):
    dfx0 = x[1]
    dfx1 = x[0]
    dfx2 = math.pi*x[2]
    return np.array([dfx0,dfx1,dfx2])

cons = ({'type': 'eq',
         'fun' : lambda x: np.array([math.pi*x[2]+2*x[0]+x[1]-12]),
         'jac' : lambda x: np.array([math.pi, 2.0, 1.0])},
        {'type': 'ineq',
         'fun' : lambda x: np.array([x[0],x[1],x[2]]),
         'jac' : lambda x: np.array([1.0, 1.0,1.0])})

x = np.array([5.,5.,2.])
#res = scipy.optimize.minimize(f_objective, x0=x , args=(), jac=func_deriv,
                #constraints=cons, method='SLSQP',options={'disp': True})

# res = scipy.optimize.minimize(f_objective, x0=x , args=(), jac=func_deriv,
#                 options={'disp': False, 'xtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None})
#
# print("Problem 1 result: ")
# print(res)

#if k=0, returns the gradient of lagrangian, if k=1, returns the hessian
def diff_L(f,x,m,k):
    #Define the lagrangian for given m and f
    L = lambda x_: f(x_)[0] + (np.matrix(f(x_)[2])*np.matrix(m).transpose())[0,0]
    return ad.gh(L)[k](x)

#Returns the gradients of the equality constraints
def grad_h(f,x):
    return  [ad.gh(lambda y:
                   f(y)[2][i])[0](x) for i in range(len(f(x)[2]))]

#Solves the quadratic problem inside the SQP method
def solve_QP(f,x,m):
    left_side_first_row = np.concatenate((\
    np.matrix(diff_L(f,x,m,1)),\
    np.matrix(grad_h(f,x)).transpose()),axis=1)
    left_side_second_row = np.concatenate((\
    np.matrix(grad_h(f,x)),\
    np.matrix(np.zeros((len(f(x)[2]),len(f(x)[2]))))),axis=1)
    right_hand_side = np.concatenate((\
    -1*np.matrix(diff_L(f,x,m,0)).transpose(),
    -np.matrix(f(x)[2]).transpose()),axis = 0)
    left_hand_side = np.concatenate((\
                                    left_side_first_row,\
                                    left_side_second_row),axis = 0)
    temp = np.linalg.solve(left_hand_side,right_hand_side)
    return temp[:len(x)],temp[len(x):]



def SQP(f,start,precision):
    x = start
    m = np.ones(len(f(x)[2]))
    f_old = float('inf')
    f_new = f(x)[0]
    while abs(f_old-f_new)>precision:
        print (x)
        f_old = f_new
        (p,v) = solve_QP(f,x,m)
        x = x+np.array(p.transpose())[0]
        m = m+v
        f_new = f(x)[0]
    return x


res = SQP(f_obj,[0,0,0],0.0001)
print(res)


#
# #Problem 2
# def rosen(x):
#     return sum(100.0*(x[1:]-x[-1]**2.0)**2.0 + (1-x[:-1])**2.0)
#
# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2, 0.5, 1.1, 0.3, 0.9, 1.1])
# res_1 = scipy.optimize.minimize(rosen, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
# res_2 = scipy.optimize.minimize(rosen, x0, method='SLSQP',options={'ftol': 1e-8, 'disp': True})
# res_3 = scipy.optimize.minimize(rosen, x0, method='CG',options={'gtol': 1e-8, 'disp': True})
#
# print("Problem 2 result: ")
# print(res_1)
# print("************************")
# print(res_2)
# print("************************")
# print(res_3)


#Problem 4
# def prob(x):
#     return [(x[0]-1)+x[1],x[0]+(x[1]-1)]
#
#
# def weighting_method_pyomo(f,w):
#     points = []
#     for wi in w:
#         model = ConcreteModel()
#         model.x = Var([0,1])
#         #weighted sum
#         model.obj = Objective(expr = wi[0]*f(model.x)[0]+wi[1]*f(model.x)[1])
#         opt = pyomo.opt.SolverFactory("ipopt") #Use ipopt
#         #Combination of expression and function
#         res=opt.solve(model) #Solve the problem
#         points.append([model.x[0].value,model.x[1].value]) #We should check for optimality...
#     return points
#
# w = np.random.random((500,2)) #500 random weights
# repr = weighting_method_pyomo(prob,w)
#
# f_repr_ws = [prob(repri) for repri in repr]
# fig = plt.figure()
# plt.scatter([z[0] for z in f_repr_ws],[z[1] for z in f_repr_ws])
# plt.show()
#
# fig = plt.figure()
# plt.scatter([x[0] for x in repr],[x[1] for x in repr])
# plt.show()

#
#
# import json
# import urllib.request, urllib.parse, urllib.error
# import ad
#
#
# def connection(x):
#     serviceUrl = "http://mhartikainen.pythonanywhere.com/evaluate/"
#     val_1 = str(x[0])
#     val_2 = str(x[1])
#     val_3 = str(x[2])
#     val_4 = str(x[3])
#     vector_val = str(val_1+"/"+val_2+"/"+val_3+"/"+val_4)
#     url = serviceUrl + vector_val
#     uh = urllib.request.urlopen(url)
#     data = uh.read()
#     try: js = json.loads(str(data, 'utf-8'))
#     except: js = None
#     print (js)
#     ieq_values = np.array(js["inequality constraint values"])
#     eq_values = np.array(js["equality constraint values"])
#     fv = float(js["function value"])
#     func_grad = np.array(js["function gradient"])
#     eq_cons_grad = np.array(js["equality constraint gradients"])
#     ieq_cons_grad = np.array(js["inequality constraint gradient"])
#     return fv, eq_values, ieq_values, func_grad, eq_cons_grad, ieq_cons_grad
#
#
# #if k=0, returns the gradient of lagrangian, if k=1, returns the hessian
# def diff_L(f,x,m,k):
#     #Define the lagrangian for given m and f
#     L = lambda x_: f(x_)[0] + (np.matrix(f(x_)[2])*np.matrix(m).transpose())[0,0]
#     return ad.gh(L)[k](x)
#
# def grad_h(f,x):
#     return f(x)[4]
#
# #Solves the quadratic problem inside the SQP method
# def solve_QP(f,x,m):
#     left_side_first_row = np.concatenate((
#     np.matrix(diff_L(f,x,m,1)),
#     np.matrix(grad_h(f,x)).transpose()),axis=1)
#     left_side_second_row = np.concatenate((
#     np.matrix(grad_h(f,x)),
#     np.matrix(np.zeros((len(f(x)[2]),len(f(x)[2]))))),axis=1)
#     right_hand_side = np.concatenate((
#     -1*np.matrix(diff_L(f,x,m,0)).transpose(),
#     -np.matrix(f(x)[2]).transpose()),axis = 0)
#     left_hand_side = np.concatenate((
#                                     left_side_first_row,
#                                     left_side_second_row),axis = 0)
#     temp = np.linalg.solve(left_hand_side,right_hand_side)
#     return temp[:len(x)],temp[len(x):]
#
# def SQP(f,start,precision):
#     x = start
#     m = np.ones(len(f(x)[2]))
#     f_old = float('inf')
#     f_new = f(x)[0]
#     while abs(f_old-f_new)>precision:
#         print (x)
#         f_old = f_new
#         (p,v) = solve_QP(f,x,m)
#         x = x+np.array(p.transpose())[0]
#         m = m+v
#         f_new = f(x)[0]
#     return x
#
#
#
# z = [1.0, 2.0, 3.0, 6.0]
#
# r = SQP(connection,z,0.00001)
#
# print(r)
#
#
#












