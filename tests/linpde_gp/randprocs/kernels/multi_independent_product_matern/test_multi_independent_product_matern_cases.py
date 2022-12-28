import numpy as np

def case_illegal_input_dim():
    return (0, 3, np.zeros(3), 3)

def case_illegal_output_dim():
    return (3, 0, np.zeros(3), 3)

def case_invalid_lengthscale_shape():
    return (2, 3, np.zeros((4, 4)), 3)

def case_illegal_p():
    return (2, 3, np.zeros((3, 2)), 4.3)