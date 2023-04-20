import numpy as np
import sympy as sp

from src.ex4_utils import kalman_step

def get_system_matrices(model: str, q: float, r: float, timestep: float = 1,
                        to_numpy: bool = True) -> tuple[np.array, np.array, np.array]:
    if(model == "random_walk"):
        # F, L and H definition
        F = [
            [0, 0],
            [0, 0]
        ]
        
        L = [
            [1, 0],
            [0, 1]
        ]
        
        H = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.float32)
    elif(model == "nearly_constant_velocity"):
        # F, L and H definition
        F = [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        L = [
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ]

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
    elif(model == "nearly_constant_acceleration"):
        # F, L and H definition
        F = [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        L = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ]

        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
    else:
        print(f"Model {model} not supported. Exiting program")
        exit()

    # Calculate Fi and Q
    T = sp.symbols('T')
    F = sp.Matrix(F)
    L = sp.Matrix(L)
    Fi = sp.exp(F*T)
    Q = sp.integrate((Fi*L)*q*(Fi*L).T, (T,0,T))

    # R is the same for all model types (observation uncertainty is not dependant on state
    # representation)
    R = r * np.array([
        [1, 0],
        [0, 1]
    ], dtype=np.float32)

    if(to_numpy):
        # Replace T values with timestep
        Fi = Fi.subs(T, timestep)
        Q = Q.subs(T, timestep)

        # Cast to numpy array
        Fi = np.array(Fi).astype(np.float32)
        Q = np.array(Q).astype(np.float32)

    return Fi, Q, H, R

def kalman_filter(x: np.array, y: np.array, model: str, q: float, r: float):
    # Get matrices
    Fi, Q, H, R = get_system_matrices(model, q, r)

    # Initialize the predicted states
    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()
    
    # Set the first state to observed state
    sx[0] = x[0]
    sy[0] = y[0]

    state=np.zeros((Fi.shape[0],1),dtype=np.float32).flatten()
    state[0]=x[0]
    state[1]=y[0]
    covariance=np.eye(Fi.shape[0],dtype=np.float32)
    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(Fi, H, Q, R,
                                              np.reshape(np.array([x[j], y[j]]), (-1, 1)),
                                              np.reshape(state,(-1, 1)), covariance)
        sx[j] = state[0]
        sy[j] = state[1]

    return sx, sy