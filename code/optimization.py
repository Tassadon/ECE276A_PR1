import jax
import jax.numpy as jnp
import quarternions

def myCostFunc(quarts,Acc,tau,omega):

    
    #print(quarts.shape[0])
    a = []
    for quart in quarts:
        a.append(quarternions.get_a(quart))
    a = jnp.array(a)
    #print(a[:,1:].shape)
    #print(Acc.shape)
    
    c2 = .5*jnp.sum(jnp.square(jnp.linalg.norm(Acc - a[:,1:],)))

    c1 = []

    for i in range(1,quarts.shape[0]):
        funko = quarternions.q_multiply(quarternions.inverse(quarts[i]),quarternions.next_quarternion(quarts[i-1],tau[i-1],omega[i-1]))
        fun = 2*quarternions.log(funko+10**-6)
        c1.append(jnp.square(jnp.linalg.norm(fun)))
    c1 = jnp.sum(jnp.array(c1))
    cost = c1 + c2

    return cost

def update_qts(quarts, gradient,alpha=.01):
    quarts = quarts + alpha*gradient
    return quarts

def optimize(quarts, Acc,tau,omega,alpha=.01):

    gradFunc = jax.jacrev(myCostFunc)
    
    for i in range(10):
        print(i)
        quarts = quarts/jnp.array([jnp.linalg.norm(quarts,axis=1)]).T
        quarts = quarts - alpha*gradFunc(quarts,Acc,tau,omega)
        print(quarts)
        print(myCostFunc(quarts,Acc,tau,omega))
        #print("Loss at iteration %d: %.2f" % (i+1, myCostFunc(dingo,Acc,tau,omega)))

    quarts = quarts/jnp.array([jnp.linalg.norm(quarts,axis=1)]).T

    return quarts

    

if __name__ == "__main__":
    joe = jnp.array([[1,2,3,4],[1,5,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],dtype=jnp.float32)
    tau = jnp.array([1,1,1,1,1])
    omega = jnp.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
    acc = jnp.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],dtype=jnp.float32)
    #print(joe/jnp.array([jnp.linalg.norm(joe,axis=1)]).T)
    #print(myCostFunc(joe,acc,tau,omega))
    subu = jax.jacrev(myCostFunc,argnums=0)
    #print(joe.shape)
    #print(tau.shape)
    #print(omega.shape)
    #print(acc.shape)
    #print(subu(joe,acc,tau,omega))
    #print(myCostFunc(joe,acc,tau,omega))
    print(optimize(joe,acc,tau,omega))