import numpy as np

def inject_adversarial_noise(stream, start, end, magnitude=1.0):
    adv_stream = stream.copy()
    perturb = magnitude * np.sign(np.random.randn(end-start, stream.shape[1]))
    adv_stream[start:end] += perturb
    return adv_stream