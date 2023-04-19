import numpy as np


def filtering(obs: np.ndarray, trans: np.ndarray, emission: np.ndarray, obs0: np.ndarray) -> np.ndarray:
    """
    The forward filtering algorithm for use in HMMs.
    Calculates P(x_T | y_1:T), i.e. the probability of each state given
    the history of states in the last T steps. Let n be the amount of classes.
    See (https://en.wikipedia.org/wiki/Forward_algorithm)

    Transition and emission probability matrix structure: 
        - Row number denotes the state you are currently are in (i.e. x_t-1)
        - Column number denotes the state that you move to (i.e. x_t)

    Args:
        obs      ((Txn) Numpy array): Observations in each of the last T frames
        trans    ((nxn) Numpy array): Transition probability of the Markov chain
        emission ((nxn) Numpy array): Emission probabilities
        obs0     ((1xn) Numpy array): Initial distribution (usually from last batch)
    Returns:
        (1xn) Numpy array of filtered class probabilities
    """
    T, n = len(obs), len(trans)
    alpha = obs0 # Initializing alpha to be the first observation

    for t in range(T):

        # Calculating alpha_t(x_t) for each instance x_t
        alpha = emission * (trans @ alpha)

        # Normalizing result:
        alpha = alpha / np.linalg.norm(alpha, ord=1, axis=1, keepdims=True)

        # Calculating weighted average based on observations
        alpha = np.sum(obs[t] * alpha[:, None], axis=0).flatten()
    
    return alpha / np.linalg.norm(alpha, ord=1)


def main() -> None:
    obs = np.array([
        [0.1, 0.9],
        [0.5, 0.5],
        [0.2, 0.8]
    ])
        
    # Defined in Section 14.3.1 on Page 492
    transition_model = np.array([[.7, .3],
                                [.3, .7]])

    # Row 1: when evidence = False
    # Row 2: when evidence = True
    sensor_model = np.array([[.1, .8],
                            [.9, .2]])
    
    obs0 = np.array([0.5, 0.5])

    print(filtering(
        obs,
        transition_model,
        sensor_model,
        obs0
    ))


if __name__ == "__main__":
    main()