import numpy as np
import pickle
from HMM.filtering import filtering


def generate_semi_uniform_transition_matrix(beta: float = 0.90, m: int = 29) -> np.ndarray:
    """
    Generates a semi-uniform transition probability matrix, i.e.
    a matrix in which the diagonal is constantly equal to beta, and
    the other values in each row are uniformly distributed such that
    it sums to 1, meaning for i != j, T[i, j] = alpha s.t.
    beta + (m-1)*alpha = 1

    Args:
        beta (float)
        m      (int): AMount of classes
    Returns:
        mxm array of inferred values
    """
    alpha = (1 - beta)/(m-1)
    transition_matrix = np.ones((m, m))*alpha
    np.fill_diagonal(transition_matrix, beta)
    return transition_matrix



""" Matrix generation"""

with open("emission_onenorm.txt", 'rb') as f:
    EMISSION = pickle.load(f)

UNIFORM_TRANSITION = np.ones((29, 29))/29

SEMI_UNIFORM_TRANSITION = generate_semi_uniform_transition_matrix(beta=0.7)



""" Inference functions """

def uniform_predict(obs: np.ndarray, obs0: np.ndarray) -> np.ndarray:
    """
    Given an array of observations, uses filtering to infer
    with a uniform transition probability matrix.
    - n: Amount of observations
    - m: Amount of classes in each observation
    
    Args:
        obs  (nxm Array): Observations
        obs0 (1xm Array): Initial value
    Returns:
        1xm array of inferred values
    """
    return filtering(obs, UNIFORM_TRANSITION, EMISSION, obs0)


def semi_uniform_predict(obs: np.ndarray, obs0: np.ndarray) -> np.ndarray:
    """
    Given an array of observations, uses filtering to infer
    with a semi-uniform transition probability matrix (see
    doc of "generate_semi_uniform_transition_matrix").
    - n: Amount of observations
    - m: Amount of classes in each observation
    
    Args:
        obs  (nxm Array): Observations
        obs0 (1xm Array): Initial value
    Returns:
        1xm array of inferred values
    """
    return filtering(obs, SEMI_UNIFORM_TRANSITION, EMISSION, obs0)