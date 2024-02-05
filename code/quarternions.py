import jax
import jax.numpy as jnp

def predict_next_quarternion(tau,omega):
    assert omega.shape[0] == 3
    q0 = jnp.array([1,0,0,0],dtype=jnp.float32)
    qt = []
    qt.append(q0)
    iterations = len(tau)

    for i in range(iterations):
        tau_t = tau[i]/2
        omega_t = omega[:,i]*tau_t
        fun = exponential( jnp.array([0,omega_t[0],omega_t[1],omega_t[2]]))
        qProduct = q_multiply(qt[-1],fun)
        qt.append(qProduct)

    return qt

def next_quarternion(q,tau,omega):
    tau_t = tau/2
    omega_t = omega*tau_t
    fun = exponential( jnp.array([0,omega_t[0],omega_t[1],omega_t[2]]))
    qProduct = q_multiply(q,fun)
    return qProduct

def q_norm(q):
    return jnp.sqrt(q[0]**2 + jnp.dot(q,q))

def q_multiply(q1,q2):
    qs = q1[0]*q2[0] - jnp.dot(q1[1:],q2[1:])
    the_cross = jnp.cross(q1[1:],q2[1:])
    qv = q1[0]*q2[1:] + q2[0]*q1[1:] + the_cross
    return jnp.array([qs, qv[0], qv[1],qv[2]],dtype=jnp.float32)

def exponential(q):
    scalar = jnp.exp(q[0])
    qs = jnp.cos(q_norm(q[1:]))
    qv = q[1:]*jnp.sin(q_norm(q[1:]))/q_norm(q[1:])
    return jnp.array([qs,qv[0],qv[1],qv[2]])*scalar

def conjugate(q):
    return jnp.array([q[0],-q[1],-q[2],-q[3]])

def inverse(q):
    return conjugate(q)/(jnp.square(q_norm(q)))

def get_a(q):

    return q_multiply(q_multiply(inverse(q),jnp.array([0,0,0,-9.81])),q)

if __name__ == "__main__":
    tau = jnp.array([[1],[1],[1],[1]])
    omega = jnp.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
    q = jnp.array([1,0,1,0])
    #print(omega.T.shape)
    G = predict_next_quarternion(tau,omega.T)
    Z = jnp.array(G)
    #print(Z.shape)
    print(get_a(q))
    