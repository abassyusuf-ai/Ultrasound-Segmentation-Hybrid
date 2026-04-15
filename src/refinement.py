import numpy as np
from skimage.segmentation import chan_vese

def refine_mask(image, predicted_mask):
    """
    Uses Chan-Vese Active Contours to refine the U-Net output.
    """
    # cv_result is the refined version of the prediction
    cv_result = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, 
                          max_num_iter=200, dt=0.5, init_level_set=predicted_mask)
    return cv_result
