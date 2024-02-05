import jax
import jax.numpy as jnp
import quarternions

def myCostFunc(quarts,Acc,tau,omega):

    a = []
    #print(quarts.shape[0])
    for quart in quarts:
        a.append(quarternions.get_a(quart))
    a = jnp.array(a)
    #print(a[:,1:].shape)
    #print(Acc.shape)
    c2 = .5*jnp.sum(jnp.square(jnp.linalg.norm(Acc - a[:,1:],)))

    c1 = []

    for i in range(1,quarts.shape[0]):
        funko = quarternions.q_multiply(quarternions.inverse(quarts[i]),quarternions.next_quarternion(quarts[i-1],tau[i-1],omega[i-1]))
        fun = 2*jnp.log(funko)
        c1.append(jnp.square(jnp.linalg.norm(fun)))
    c1 = jnp.sum(jnp.array(c1))
    cost = c1 + c2

    return cost

def update_qts(quarts, gradient,alpha=.1):
    quarts = quarts + alpha*gradient
    return quarts

def optimize(quarts, Acc,tau,omega,alpha=.1):

    gradFunc = jax.jacfwd(myCostFunc)
    
    for i in range(10):
        quarts = quarts/jnp.array([jnp.linalg.norm(quarts,axis=1)]).T
        gradient = gradFunc(quarts,Acc,tau,omega)
        quarts = update_qts(quarts, gradient,alpha=alpha)
        
        print("Loss at iteration %d: %.2f" % (i+1, myCostFunc(quarts,Acc,tau,omega)))

    return quarts



if __name__ == "__main__":
    joe = jnp.array([[1,2,3,4],[1,5,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],dtype=jnp.float32)
    tau = jnp.array([1,1,1,1,1])
    omega = jnp.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
    acc = jnp.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],dtype=jnp.float32)
    #print(joe/jnp.array([jnp.linalg.norm(joe,axis=1)]).T)
    #print(myCostFunc(joe,acc,tau,omega))
    subu = jax.jacfwd(myCostFunc,argnums=0)
    print(joe.shape)
    print(tau.shape)
    print(omega.shape)
    print(acc.shape)
    #print(subu(joe,acc,tau,omega))

    optimize(joe,acc,tau,omega)