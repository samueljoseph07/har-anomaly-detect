import numpy as np

def sliding_windows(stream, window_size, step):
    for start in range(0, len(stream) - window_size + 1, step):
        yield stream[start:start+window_size]