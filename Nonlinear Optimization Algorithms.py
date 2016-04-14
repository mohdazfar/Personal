#COURSE LINK: https://github.com/maeehart/TIES483

import math
#######Bisection Method#########

def bisection_line_search(a,b,f,L,epsilon):
    x = a
    y = b
    while y-x>2*L:
        if f((x+y)/2+epsilon)>f((x+y)/2-epsilon):
            y=(x+y)/2+epsilon
        else:
            x = (x+y)/2-epsilon
    return (x+y)/2



######## Golden Section Search ###########
def golden_section_line_search(a,b,f,L):
    x = a
    y = b
    while y-x>2*L:
        if f(x+(math.sqrt(5.0)-1)/2.0*(y-x))<f(y-(math.sqrt(5.0)-1)/2.0*(y-x)):
            x = y-(math.sqrt(5.0)-1)/2.0*(y-x)
        else:
            y = x+(math.sqrt(5.0)-1)/2.0*(y-x)
    return (x+y)/2

####### Gradient Visualisation ##########
def f_simple(x):
    return (x[0] - 10.0)**2 + (x[1] + 5.0)**2+x[0]**2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pylab import meshgrid
def visualize_gradient(f,point,x_lim,y_lim):
    grad_point = np.array(ad.gh(f)[0](point))
    grad_point = grad_point/np.linalg.norm(grad_point)
    X,Y,Z = point[0],point[1],f(point)
    U,V,W = grad_point[0],grad_point[1],0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(x_lim[0],x_lim[1],0.1)
    y = np.arange(y_lim[0],y_lim[1],0.1)
    X2,Y2 = meshgrid(x, y) # grid of point
    Z2 = [f([x,y]) for (x,y) in zip (X2,Y2)] # evaluation of the function on the grid
    surf = ax.plot_surface(X2, Y2, Z2,alpha=0.5)
    ax.quiver(X,Y,Z,U,V,W,color='red',linewidth=1.5)
    return plt


######## Steepest Descent algorithm for unconstrained optimization ########
import numpy as np
import ad
def steepest_descent(f,start,step,precision):
    f_old = float('Inf')
    x = np.array(start)
    steps = []
    f_new = f(x)
    while abs(f_old-f_new)>precision:
        f_old = f_new
        d = -np.array(ad.gh(f)[0](x))
        x = x+d*step
        f_new = f(x)
        steps.append(list(x))
    return x,f_new,steps


###Plot steps of steepest descent
import matplotlib.pyplot as plt

def plot_2d_steps(steps,start):
    myvec = np.array([start]+steps).transpose()
    plt.plot(myvec[0,],myvec[1,],'ro')
    for label,x,y in zip([str(i) for i in range(len(steps)+1)],myvec[0,],myvec[1,]):
        plt.annotate(label,xy = (x, y))
    return plt


####### Newton's method ##########
def newton(f,start,step,precision):
    f_old = float('Inf')
    x = np.array(start)
    steps = []
    f_new = f(x)
    while abs(f_old-f_new)>precision:
        f_old = f_new
        H_inv = np.linalg.inv(np.matrix(ad.gh(f)[1](x)))
        d = (-H_inv*(np.matrix(ad.gh(f)[0](x)).transpose())).transpose()
        #Change the type from np.matrix to np.array so that we can use it in our function
        x = np.array(x+d*step)[0]
        f_new = f(x)
        steps.append(list(x))
    return x,f_new,steps


########### Penalty function methods ############
import numpy as np
def f_constrained(x):
    return np.linalg.norm(x)**2,[x[0]+x[1]-1],[]

def alpha(x,f):
    (_,ieq,eq) = f(x)
    return sum([min([0,ieq_j])**2 for ieq_j in ieq])+sum([eq_k**2 for eq_k in eq])

def penalized_function(x,f,r):
    return f(x)[0] + r*alpha(x,f)

from scipy.optimize import minimize
res = minimize(lambda x:penalized_function(x,f_constrained,100000),
               [0,0],method='Nelder-Mead',
         options={'disp': True})
print (res.x)


(f_val,ieq,eq) = f_constrained(res.x)
print ("Value of f is "+str(f_val))
if len(ieq)>0:
    print ("The values of inequality constraints are:")
    for ieq_j in ieq:
        print (str(ieq_j)+", ")
if len(eq)>0:
    print ("The values of the equality constraints are:")
    for eq_k in eq:
        print (str(eq_k)+", ")

if all([ieq_j>=0 for ieq_j in ieq]) and all([eq_k==0 for eq_k in eq]):
    print ("Solution is feasible")
else:
    print ("Solution is infeasible")


######## Barrier function method ##########
def beta(x,f):
    _,ieq,_ = f(x)
    try:
        value=sum([1/max([0,ieq_j]) for ieq_j in ieq])
    except ZeroDivisionError:
        value = float("inf")
    return value

def function_with_barrier(x,f,r):
    return f(x)[0]+r*beta(x,f)

from scipy.optimize import minimize
res = minimize(lambda x:function_with_barrier(x,f_constrained,0.00000000000001),
               [1,1],method='Nelder-Mead', options={'disp': True})
print (res.x)

(f_val,ieq,eq) = f_constrained(res.x)
print ("Value of f is "+str(f_val))
if len(ieq)>0:
    print ("The values of inequality constraints are:")
    for ieq_j in ieq:
        print (str(ieq_j)+", ")
if len(eq)>0:
    print ("The values of the equality constraints are:")
    for eq_k in eq:
        print (str(eq_k)+", ")
if all([ieq_j>=0 for ieq_j in ieq]) and all([eq_k==0 for eq_k in eq]):
    print ("Solution is feasible")
else:
    print ("Solution is infeasible")


######### Projected Gradient ############33
import numpy as np
def project_vector(A,vector):
    #convert A into a matrix
    A_matrix = np.matrix(A)
    #construct the "first row" of the matrix [[I,A^T],[A,0]]
    left_matrix_first_row = np.concatenate((np.identity(len(vector)),A_matrix.transpose()), axis=1)
    #construct the "second row" of the matrix
    left_matrix_second_row = np.concatenate((A_matrix,np.matrix(np.zeros([len(A),len(vector)+len(A)-len(A[0])]))), axis=1)
    #combine the whole matrix by combining the rows
    left_matrix = np.concatenate((left_matrix_first_row,left_matrix_second_row),axis = 0)
    #Solve the system of linear equalities from the previous page
    return np.linalg.solve(left_matrix, \
                           np.concatenate((np.matrix(vector).transpose(),\
                                           np.zeros([len(A),1])),axis=0))[:len(vector)]


import ad
def projected_gradient_method(f,A,start,step,precision):
    f_old = float('Inf')
    x = np.array(start)
    steps = []
    f_new = f(x)
    while abs(f_old-f_new)>precision:
        f_old = f_new
        gradient = ad.gh(f)[0](x)
        grad_proj = project_vector(A,[-i for i in gradient])#The only changes to steepest..
        grad_proj = np.array(grad_proj.transpose())[0] #... descent are here!
#        import pdb; pdb.set_trace()
        x = x+grad_proj*step
        f_new = f(x)
        steps.append(list(x))
    return x,f_new,steps


############# Sequential Quadratic Programming (SQP) ##################

import numpy as np
import ad



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

