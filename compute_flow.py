import numpy as np

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    # Initialize arrays to hold the derivatives
    A = np.zeros((size**2, 2))
    b = np.zeros((size**2, 1))

    # Calculate half the size
    half_size = size // 2

    # Counter for the equations
    eq = 0

    # Iterate over the patch
    for i in range(-half_size, half_size + 1):
        for j in range(-half_size, half_size + 1):
            # Ensure we are within the image boundaries
            if x+j < 0 or x+j >= Ix.shape[1] or y+i < 0 or y+i >= Ix.shape[0]:
                continue
            A[eq] = [Ix[y+i, x+j], Iy[y+i, x+j]]
            b[eq] = -It[y+i, x+j]
            eq += 1

    # Use np.linalg.lstsq to solve for flow (u, v)
    flow, _, _, smin = np.linalg.lstsq(A[:eq], b[:eq], rcond=-1)

    # The smallest singular value is our confidence
    conf = smin[-1]

    return flow.flatten(), conf

def flow_lk(Ix, Iy, It, size=5):
    H, W = Ix.shape
    image_flow = np.zeros((H, W, 2))
    confidence = np.zeros((H, W))

    # Iterate over all patches
    for y in range(H):
        for x in range(W):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y, size)
            image_flow[y, x] = flow
            confidence[y, x] = conf

    return image_flow, confidence
