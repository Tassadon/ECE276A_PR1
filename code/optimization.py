import autograd.numpy as np
from autograd import grad
import quarternions
from transforms3d import euler

def myCostFunc(quarts,Acc,tau,omega):
    
    for i,quart in enumerate(quarts):
        if i == 0:
            a = quarternions.get_a(quart)
        else:
            a = np.vstack((a,quarternions.get_a(quart)))

    c2 = .5*np.sum(np.square(np.linalg.norm(Acc - a[:,1:])))

    for i in range(1,quarts.shape[0]):
        funko = quarternions.q_multiply(quarternions.inverse(quarts[i]),quarternions.next_quarternion(quarts[i-1],tau[i-1],omega[i-1]))
        fun = 2*quarternions.log(funko)
        if i == 1:
            c1 = np.array([np.square(np.linalg.norm(fun))])
        else:
            c1 = np.vstack((c1,np.array([np.square(np.linalg.norm(fun))])))

    c1 = .5*np.sum(np.array(c1))
    cost = c1 + c2

    return cost

def optimize(quarts, Acc,tau,omega,alpha=.01,epochs=10):
    assert type(epochs) is int
    gradFunc = grad(myCostFunc,0)
    
    for i in range(epochs):
        quarts = quarts/np.array([np.linalg.norm(quarts,axis=1)]).T
        quarts = quarts - alpha*gradFunc(quarts,Acc,tau,omega)
        print("Loss at iteration %d: %.2f" % (i+1, myCostFunc(quarts,Acc,tau,omega)))

    quarts = quarts/np.array([np.linalg.norm(quarts,axis=1)]).T

    return quarts


    

if __name__ == "__main__":
    joe = np.array([[1,2,3,4],[1,5,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],dtype=np.float32)
    tau = np.array([1,1,1,1,1])
    omega = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
    acc = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],dtype=np.float32)
    #print(joe/np.array([np.linalg.norm(joe,axis=1)]).T)
    #print(myCostFunc(joe,acc,tau,omega))
    subu = grad(myCostFunc,0)
    sub = subu(joe,acc,tau,omega)
    print(sub.shape)
    #print(joe.shape)
    #print(tau.shape)
    #print(omega.shape)
    #print(acc.shape)
    #print(subu(joe,acc,tau,omega))
    #print(myCostFunc(joe,acc,tau,omega))
    A = optimize(joe,acc,tau,omega,alpha=.01,epochs=100)
    print(A)
    #print(np.linalg.norm(A,axis=1))