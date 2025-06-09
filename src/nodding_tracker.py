import numpy as np
from scipy.signal import find_peaks

"""
Naive nodding detector uses head pitch time series
to detect up and downs of the head.
"""
class NoddingDetector:
    def __init__(self):
        pass
    
    def detect_nodding(self, head_pitch, prominence = 2):
        # negative to make it clockwise
        head_pitch = -np.array(head_pitch)
        
        # find local minima and maxima
        # they will represent head up and downs
        mins, _ = find_peaks(-head_pitch, prominence=prominence)
        maxs, _ = find_peaks(head_pitch, prominence=prominence)
        
        # lets assume that the first point is local max and last is local min
        max_queue = list(maxs)
        min_queue = list(mins)
        if max_queue[0] > min_queue[0]:
            max_queue.insert(0, 0)
        if min_queue[-1] < max_queue[-1]:
            min_queue.append(len(head_pitch) - 1)

        # now just count the pairs
        nodding_interv = []
        while len(max_queue) > 0 and len(min_queue) > 0:
            start_pos = max_queue.pop(0)
            down_pos = min_queue.pop(0)
            nodding_interv.append((start_pos, down_pos))
            
        return nodding_interv