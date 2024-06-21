import numpy as np
import matplotlib.pyplot as plt

def epipole(flow_x, flow_y, smin, thresh, num_iterations=None):
    """
    Compute the epipole from the flows,
    Inputs:
    - flow_x: optical flow on the x-direction - shape: (H, W)
    - flow_y: optical flow on the y-direction - shape: (H, W)
    - smin: confidence of the flow estimates - shape: (H, W)
    - thresh: threshold for confidence - scalar
    - Ignore num_iterations
    Outputs:
    - ep: epipole - shape: (3,)
    """
    # Logic to compute the points you should use for your estimation
    # We only look at image points above the threshold in our image
    # Due to memory constraints, we cannot use all points on the autograder
    # Hence, we give you valid_idx which are the flattened indices of points
    # to use in the estimation estimation problem
    good_idx = np.flatnonzero(smin > thresh)
    permuted_indices = np.random.RandomState(seed=10).permutation(good_idx)
    valid_idx = permuted_indices[:3000]
    
    ### STUDENT CODE START - PART 1 ###
    # 1. For every pair of valid points, compute the epipolar line (use np.cross)
    # Hint: for faster computation and more readable code, avoid for loops! Use vectorized code instead.
    
    h, w = flow_x.shape
    x, y = np.meshgrid(np.arange(-256, 256), np.arange(-256, 256))
    x, y = x.reshape(h, w), y.reshape(h, w)
    
    u, v, x, y = flow_x.flatten()[valid_idx], flow_y.flatten()[valid_idx], x.flatten()[valid_idx], y.flatten()[valid_idx]
    
    U = np.stack((u, v, np.zeros_like(u)), axis=1)
    Xp = np.stack((x, y, np.ones_like(x)), axis=1)
    xpu = np.cross(Xp, U)
    
    # 2. Solve the epipole either use SVD or lstsq
    _, _, Vt = np.linalg.svd(xpu)
    ep = Vt[-1]
    
    ### STUDENT CODE END - PART 1 ###
    
    return ep