import autograd.numpy as np
from autograd import grad

def predict_next_quarternion(tau,omega):
    assert omega.shape[0] == 3
    qt = np.zeros((omega.shape[1],4))
    qt[0] = np.array([1,0,0,0])
    iterations = len(tau)

    for i in range(iterations):
        tau_t = tau[i]/2
        omega_t = omega[:,i]*tau_t
        fun = exponential( np.array([0,omega_t[0],omega_t[1],omega_t[2]]))
        qProduct = q_multiply(qt[i],fun)
        qt[i+1] = qProduct

    return qt


def next_quarternion(q,tau,omega):
    tau_t = tau/2
    omega_t = omega*tau_t
    fun = exponential( np.array([0,omega_t[0],omega_t[1],omega_t[2]]))
    qProduct = q_multiply(q,fun)
    return qProduct


def q_norm(q):
    return np.sqrt(q[0]**2 + np.dot(q+10**-8,q+10**-8))


def q_multiply(q1,q2):

    qs = q1[0]*q2[0] - np.dot(q1[1:],q2[1:])
    the_cross = np.cross(q1[1:],q2[1:])
    qv = q1[0]*q2[1:] + q2[0]*q1[1:] + the_cross
    return np.array([qs, qv[0], qv[1],qv[2]],dtype=np.float32)


def exponential(q):
    scalar = np.exp(q[0])
    qs = np.cos(q_norm(q[1:]))
    qv = q[1:]*np.sin(q_norm(q[1:]))/q_norm(q[1:])
    return np.array([qs,qv[0],qv[1],qv[2]])*scalar


def conjugate(q):
    return np.array([q[0],-q[1],-q[2],-q[3]])


def inverse(q):
    return conjugate(q)/(np.square(q_norm(q)))


def get_a(q):
    g = -10
    return q_multiply(q_multiply(inverse(q),np.array([0,0,0,-g])),q)

def log(q):
    qv = q[1:]*np.arccos(q[0]/q_norm(q))/q_norm(q)
    DOOBO = np.array([np.log(q_norm(q)),qv[0],qv[1],qv[2]])
    return DOOBO

if __name__ == "__main__":
    tau = np.array([[1],[1],[1],[1]])
    omega = np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
    q = np.array([1,0,0,0])
    #print(omega.T.shape)
    G = predict_next_quarternion(tau,omega.T)
    Z = np.array(G)
    #print(Z.shape)
    #print(get_a(q))
    print(G)