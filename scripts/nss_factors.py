import numpy as np

def x_1(t : float, lambda_1 : float) -> float:
    if t==0:
        return 1.0
    return (1-np.exp(-(t/lambda_1)))/(t/lambda_1)

def x_2(t : float, lambda_1 : float) -> float:
    if t==0:
        return 0.0
    return (1-np.exp(-t/lambda_1))/(t/lambda_1)-np.exp(-t/lambda_1)

def x_3(t : float, lambda_2 : float) -> float:
    if t==0:
        return 0.0
    return (1-np.exp(-t/lambda_2))/(t/lambda_2)-np.exp(-t/lambda_2)

def build_factor_loadings_matrix(time_list : list, is_ns : bool, lambda_1 : float, lambda_2 : float|None) -> np.ndarray:
    """
    Design matrix of NS/NSS factor loadings (level, slope, curvature[, second
    curvature]) evaluated at each maturity in time_list, for fixed lambda(s).
    """
    
    if is_ns:
        factor_loadings_matrix=np.array([[1, x_1(t, lambda_1), x_2(t, lambda_1)] for t in time_list])

    else:
        factor_loadings_matrix=np.array([[1, x_1(t, lambda_1), x_2(t, lambda_1), x_3(t, lambda_2)] for t in time_list])

    return factor_loadings_matrix

def unpack_scipy_result(scipy_result, is_ns: bool) -> tuple[np.ndarray, float, float | None]:
    """
    Splits a scipy OptimizeResult from sp_calibration into (beta, lambda_1,
    lambda_2), following the [betas..., lambda(s)] parameter ordering used
    by sp_calibration's starting point.
    """
    parameters = scipy_result.x

    if is_ns:
        return parameters[:3], parameters[3], None
    return parameters[:4], parameters[4], parameters[5]