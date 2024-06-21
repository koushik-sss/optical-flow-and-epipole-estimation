import numpy as np
import matplotlib.pyplot as plt

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    Plot a flow field of one frame of the data.
    
    Inputs:
        - image: grayscale image - shape: (H, W)
        - flow_image: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - threshmin: threshold for confidence (optional) - scalar
    """
    
    H, W = image.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    valid_mask = confidence > threshmin
    flow_x = flow_image[valid_mask, 0]
    flow_y = flow_image[valid_mask, 1]
    x = x[valid_mask]
    y = y[valid_mask]

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.quiver(x, y, flow_x, flow_y, color='red', scale=30, width=0.002)
    plt.show()
    
    return