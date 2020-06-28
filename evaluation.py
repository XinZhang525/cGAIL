import numpy as np

def cross_entropy(target, ground_truth): # actually the KL-divergence
    epsilon = 1e-12
    ce = 0.
    target = target.copy()
    ground_truth = ground_truth.copy()
    ces = []
    ce2s = []
    for state in range(len(ground_truth)):
        target_prime = np.clip(target[state], epsilon, 1.-epsilon)
        ground_truth[state] = np.clip(ground_truth[state], epsilon, 1.-epsilon)
        t = np.sum(ground_truth[state]*np.log((target_prime/ground_truth[state])))
        ce -= t
        ces.append(t)
        
    return ce/(len(target)), ces
