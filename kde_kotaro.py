from scipy.stats import multivariate_normal

def kernel_density_estimation(data, kernel_widths):
    """
    Perform Kernel Density Estimation (KDE) on the data.
    
    Arguments:
    data: An (n, d) numpy array, where n is the number of data points and d is the dimensionality.
    kernel_widths: A list or numpy array of length n, giving the kernel width for each data point.
    
    Returns:
    A function representing the KDE.
    """
    assert len(data) == len(kernel_widths), "Data and kernel widths must have the same length."

    def kde(x):
        """
        Evaluate the KDE at a given point x.
        
        Arguments:
        x: A numpy array of length d.
        
        Returns:
        The KDE evaluated at x.
        """
        result = 0
        for data_point, width in zip(data, kernel_widths):
            kernel = multivariate_normal(mean=data_point, cov=width**2)
            result += kernel.pdf(x)
        return result / len(data)
    
    return kde