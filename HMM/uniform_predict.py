import numpy as np
import pickle
from HMM.filtering import filtering

with open("emission.txt", 'rb') as f:
    emission = pickle.load(f)
print(emission.shape)
transition = np.ones((29, 29))/29

def uniform_predict(obs: np.ndarray, obs0: np.ndarray):
    return filtering(obs, transition, emission, obs0)