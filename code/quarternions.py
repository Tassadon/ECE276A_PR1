import jax
import jax.numpy as jnp

def predict_next_quarternion(tau,omega):
    q0 = jnp.array([1,0,0,0],dtype=jnp.float32)
    qt = []
    qt.append(q0)
    iterations = len(tau)
    for i in range(iterations):
        tau_t = tau[i]
        omega_t = omega[:,i]
        fun = exponential( jnp.array([tau_t[0],omega_t[0],omega_t[1],omega_t[2]]))
        qProduct = q_multiply(qt[-1],fun)
        qt.append(qProduct)

    return qt

def q_multiply(q1,q2):
    qs = q1[0]*q2[0] - jnp.dot(q1,q2)
    the_cross = jnp.cross(q1[1:],q2[1:])
    qv = q1[0]*q2[1:] + q2[0]*q1[1:] + the_cross
    return jnp.array([qs, qv[0], qv[1],qv[2]],dtype=jnp.float32)

def exponential(q):
    scalar = jnp.exp(q[0])
    qs = jnp.cos(jnp.linalg.norm(q[1:]))
    qv = q[1:]/jnp.linalg.norm(q[1:])*jnp.sin(jnp.linalg.norm(q[1:]))
    return jnp.array([qs,qv[0],qv[1],qv[2]])


def yaw_pitch_roll(q):
    """Get the equivalent yaw-pitch-roll angles aka. intrinsic Tait-Bryan angles following the z-y'-x'' convention

    Returns:
        yaw:    rotation angle around the z-axis in radians, in the range `[-pi, pi]`
        pitch:  rotation angle around the y'-axis in radians, in the range `[-pi/2, pi/2]`
        roll:   rotation angle around the x''-axis in radians, in the range `[-pi, pi]` 

    The resulting rotation_matrix would be R = R_x(roll) R_y(pitch) R_z(yaw)

    Note: 
        This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
    """
    q = q/jnp.linalg.norm(q)
    qs = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]


    #print(2*(qx*qz-qs*qy), q)
    if 2*(qx*qz-qs*qy)>=0.94: #Preventing gimbal lock for north pole
        yaw = jnp.arctan2(qx*qy-qs*qz,qx*qz+qs*qy)
        roll = 0
    elif 2*(qx*qz-qs*qy)<=-0.94: #Preventing gimbal lock for south pole
        yaw = -jnp.arctan2(qx*qy-qs*qz,qx*qz+qs*qy)
        roll = 0
    else:
        yaw = jnp.arctan2(qy*qz + qs*qx,
            1/2 - (qx**2 + qy**2))
        roll = jnp.arctan2(qx*qy - qs*qz,
            1/2 - (qy**2 + qz**2))
    pitch = jnp.arcsin(-2*(qx * qz - qs * qy))

    return [yaw, pitch, roll]

if __name__ == "__main__":
    tau = jnp.array([[.1],[.1],[.1],[.1]])
    omega = jnp.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
    q = jnp.array([1,0,1,0])
    [yaw, pitch,roll] = yaw_pitch_roll(q)
    print(yaw,pitch,roll)
    predict_next_quarternion(tau,omega)