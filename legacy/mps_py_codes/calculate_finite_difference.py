import numpy as np



def central_dif(matrix, axis, dx, order=4): #boundary periodic and all orders and all axis 
    """
    Calculate the derivative of a 3D matrix along a specified axis using a finite difference scheme
    with periodic boundary conditions.

    Parameters:
    - matrix: np.array, the 3D input matrix (shape: Nx * Ny * Nz).
    - axis: int, the axis along which to compute the derivative (0, 1, or 2).
    - dx: float, the spacing between points along the given axis.
    - order: int, the finite difference order (2, 4, 6, or 8).

    Returns:
    - derivative: np.array, the 3D matrix of derivatives with the same shape as input.
    """
    if order == 2:
        # 2nd-order central finite difference scheme:
        # Approximation: f'(x) ≈ (f(x+dx) - f(x-dx)) / (2dx)
        coeffs = np.array([-1, 0, 1]) / (2 * dx)

        derivative = (
            coeffs[2] * np.roll(matrix, -1, axis=axis) +  # f(x+dx)
            coeffs[0] * np.roll(matrix, 1, axis=axis)      # f(x-dx)
        )

    elif order == 4:
        # 4th-order central finite difference scheme:
        # Approximation: f'(x) ≈ (-f(x-2dx) + 8f(x-dx) - 8f(x+dx) + f(x+2dx)) / (12dx)
        coeffs = np.array([1, -8, 0, 8, -1]) / (12 * dx)

        derivative = (
            coeffs[4] * np.roll(matrix, -2, axis=axis) +  # f(x-2dx)
            coeffs[3] * np.roll(matrix, -1, axis=axis) +  # f(x-dx)
            coeffs[1] * np.roll(matrix, 1, axis=axis) +   # f(x+dx)
            coeffs[0] * np.roll(matrix, 2, axis=axis)     # f(x+2dx)
        )

    elif order == 6:
        # 6th-order central finite difference scheme:
        # Approximation: f'(x) ≈ (-f(x-3dx) + 9f(x-2dx) - 45f(x-dx) + 45f(x+dx) - 9f(x+2dx) + f(x+3dx)) / (60dx)
        coeffs = np.array([-1, 9, -45, 0, 45, -9, 1]) / (60 * dx)

        derivative = (
            coeffs[6] * np.roll(matrix, -3, axis=axis) +  # f(x-3dx)
            coeffs[5] * np.roll(matrix, -2, axis=axis) +  # f(x-2dx)
            coeffs[4] * np.roll(matrix, -1, axis=axis) +  # f(x-dx)
            coeffs[2] * np.roll(matrix, 1, axis=axis) +   # f(x+dx)
            coeffs[1] * np.roll(matrix, 2, axis=axis) +   # f(x+2dx)
            coeffs[0] * np.roll(matrix, 3, axis=axis)     # f(x+3dx)
        )

    elif order == 8:
        # 8th-order central finite difference scheme:
        # Approximation: f'(x) ≈ (3f(x-4dx) - 32f(x-3dx) + 168f(x-2dx) - 672f(x-dx) + 0f(x) + 672f(x+dx) - 168f(x+2dx) + 32f(x+3dx) - 3f(x+4dx)) / (840dx)
        coeffs = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / (840 * dx)

        derivative = (
            coeffs[8] * np.roll(matrix, -4, axis=axis) +  # f(x-4dx)
            coeffs[7] * np.roll(matrix, -3, axis=axis) +  # f(x-3dx)
            coeffs[6] * np.roll(matrix, -2, axis=axis) +  # f(x-2dx)
            coeffs[5] * np.roll(matrix, -1, axis=axis) +  # f(x-dx)
            coeffs[3] * np.roll(matrix, 1, axis=axis) +   # f(x+dx)
            coeffs[2] * np.roll(matrix, 2, axis=axis) +   # f(x+2dx)
            coeffs[1] * np.roll(matrix, 3, axis=axis) +   # f(x+3dx)
            coeffs[0] * np.roll(matrix, 4, axis=axis)     # f(x+4dx)
        )

    else:
        raise ValueError("Order must be 2, 4, 6, or 8.")

    return derivative


def central_dif_3d(matrix, axis, dx): #boundaries backward and forward
    """
    Calculate the derivative of a 3D matrix along a specified axis using central difference.

    Parameters:
    - matrix: np.array, the 3D input matrix (shape: Nx * Ny * Nz).
    - axis: int, the axis along which to compute the derivative (0, 1, or 2).
    - dx: float, the spacing between points along the given axis.

    Returns:
    - derivative: np.array, the 3D matrix of derivatives with the same shape as input.
    """
    derivative = np.zeros_like(matrix)

    # Calculate central difference along the specified axis
    if axis == 0:  # Along x-direction
        derivative[1:-1, :, :] = (matrix[2:, :, :] - matrix[:-2, :, :]) / (2 * dx)
        # Forward difference at the boundaries
        derivative[0, :, :] = (matrix[1, :, :] - matrix[0, :, :]) / dx
        derivative[-1, :, :] = (matrix[-1, :, :] - matrix[-2, :, :]) / dx

    elif axis == 1:  # Along y-direction
        derivative[:, 1:-1, :] = (matrix[:, 2:, :] - matrix[:, :-2, :]) / (2 * dx)
        # Forward difference at the boundaries
        derivative[:, 0, :] = (matrix[:, 1, :] - matrix[:, 0, :]) / dx
        derivative[:, -1, :] = (matrix[:, -1, :] - matrix[:, -2, :]) / dx

    elif axis == 2:  # Along z-direction
        derivative[:, :, 1:-1] = (matrix[:, :, 2:] - matrix[:, :, :-2]) / (2 * dx)
        # Forward difference at the boundaries
        derivative[:, :, 0] = (matrix[:, :, 1] - matrix[:, :, 0]) / dx
        derivative[:, :, -1] = (matrix[:, :, -1] - matrix[:, :, -2]) / dx

    return derivative



def finite_dif_4th_order_zero(matrix, axis, dx): #zero boundary conditions
    """
    Calculate the derivative of a 3D matrix along a specified axis using a 4th order finite difference scheme.

    Parameters:
    - matrix: np.array, the 3D input matrix (shape: Nx * Ny * Nz).
    - axis: int, the axis along which to compute the derivative (0, 1, or 2).
    - dx: float, the spacing between points along the given axis.

    Returns:
    - derivative: np.array, the 3D matrix of derivatives with the same shape as input.
    """
    # Initialize the derivative array
    derivative = np.zeros_like(matrix)
    
    # Apply the 4th order centered finite difference scheme for interior points
    if axis == 0:  # x-axis
        derivative[2:-2, :, :] = (matrix[:-4, :, :] - 8 * matrix[1:-3, :, :] + 8 * matrix[3:-1, :, :] - matrix[4:, :, :]) / (12 * dx)
    elif axis == 1:  # y-axis
        derivative[:, 2:-2, :] = (matrix[:, :-4, :] - 8 * matrix[:, 1:-3, :] + 8 * matrix[:, 3:-1, :] - matrix[:, 4:, :]) / (12 * dx)
    elif axis == 2:  # z-axis
        derivative[:, :, 2:-2] = (matrix[:, :, :-4] - 8 * matrix[:, :, 1:-3] + 8 * matrix[:, :, 3:-1] - matrix[:, :, 4:]) / (12 * dx)

    # Handle boundary points if necessary (here, we simply set them to zero, or you could apply lower-order schemes)
    derivative[:2, :, :] = 0  # Adjust for axis 0 boundary points
    derivative[-2:, :, :] = 0
    derivative[:, :2, :] = 0  # Adjust for axis 1 boundary points
    derivative[:, -2:, :] = 0
    derivative[:, :, :2] = 0  # Adjust for axis 2 boundary points
    derivative[:, :, -2:] = 0

    return derivative



def finite_dif_4th_order_periodic(matrix, axis, dx): #periodic boundary conditions 
    """
    Calculate the derivative of a 3D matrix along a specified axis using a 4th order finite difference scheme
    with periodic boundary conditions. Central difference is applied for the boundary points.

    Parameters:
    - matrix: np.array, the 3D input matrix (shape: Nx * Ny * Nz).
    - axis: int, the axis along which to compute the derivative (0, 1, or 2).
    - dx: float, the spacing between points along the given axis.

    Returns:
    - derivative: np.array, the 3D matrix of derivatives with the same shape as input.
    """
    # Initialize the derivative array
    derivative = np.zeros_like(matrix)
    Nx, Ny, Nz = matrix.shape

    # Apply the 4th order centered finite difference scheme for interior points
    if axis == 0:  # x-axis
        derivative[2:-2, :, :] = (
            matrix[:-4, :, :] - 
            8 * matrix[1:-3, :, :] + 
            8 * matrix[3:-1, :, :] - 
            matrix[4:, :, :]
        ) / (12 * dx)
        
        # Handle boundary points using periodic boundary conditions
        derivative[0, :, :] = (
            matrix[-2, :, :] - 
            8 * matrix[-1, :, :] + 
            8 * matrix[1, :, :] - 
            matrix[2, :, :]
        ) / (12 * dx)
        
        derivative[1, :, :] = (
            matrix[-1, :, :] - 
            8 * matrix[0, :, :] + 
            8 * matrix[2, :, :] - 
            matrix[3, :, :]
        ) / (12 * dx)
        
        derivative[-1, :, :] = (
            matrix[-3, :, :] - 
            8 * matrix[-2, :, :] + 
            8 * matrix[0, :, :] - 
            matrix[1, :, :]
        ) / (12 * dx)
        
        derivative[-2, :, :] = (
            matrix[-4, :, :] - 
            8 * matrix[-3, :, :] + 
            8 * matrix[-1, :, :] - 
            matrix[0, :, :]
        ) / (12 * dx)

    elif axis == 1:  # y-axis
        derivative[:, 2:-2, :] = (
            matrix[:, :-4, :] - 
            8 * matrix[:, 1:-3, :] + 
            8 * matrix[:, 3:-1, :] - 
            matrix[:, 4:, :]
        ) / (12 * dx)

        # Handle boundary points using periodic boundary conditions
        derivative[:, 0, :] = (
            matrix[:, -2, :] - 
            8 * matrix[:, -1, :] + 
            8 * matrix[:, 1, :] - 
            matrix[:, 2, :]
        ) / (12 * dx)
        
        derivative[:, 1, :] = (
            matrix[:, -1, :] - 
            8 * matrix[:, 0, :] + 
            8 * matrix[:, 2, :] - 
            matrix[:, 3, :]
        ) / (12 * dx)
        
        derivative[:, -1, :] = (
            matrix[:, -3, :] - 
            8 * matrix[:, -2, :] + 
            8 * matrix[:, 0, :] - 
            matrix[:, 1, :]
        ) / (12 * dx)
        
        derivative[:, -2, :] = (
            matrix[:, -4, :] - 
            8 * matrix[:, -3, :] + 
            8 * matrix[:, -1, :] - 
            matrix[:, 0, :]
        ) / (12 * dx)

    elif axis == 2:  # z-axis
        derivative[:, :, 2:-2] = (
            matrix[:, :, :-4] - 
            8 * matrix[:, :, 1:-3] + 
            8 * matrix[:, :, 3:-1] - 
            matrix[:, :, 4:]
        ) / (12 * dx)

        # Handle boundary points using periodic boundary conditions
        derivative[:, :, 0] = (
            matrix[:, :, -2] - 
            8 * matrix[:, :, -1] + 
            8 * matrix[:, :, 1] - 
            matrix[:, :, 2]
        ) / (12 * dx)
        
        derivative[:, :, 1] = (
            matrix[:, :, -1] - 
            8 * matrix[:, :, 0] + 
            8 * matrix[:, :, 2] - 
            matrix[:, :, 3]
        ) / (12 * dx)
        
        derivative[:, :, -1] = (
            matrix[:, :, -3] - 
            8 * matrix[:, :, -2] + 
            8 * matrix[:, :, 0] - 
            matrix[:, :, 1]
        ) / (12 * dx)
        
        derivative[:, :, -2] = (
            matrix[:, :, -4] - 
            8 * matrix[:, :, -3] + 
            8 * matrix[:, :, -1] - 
            matrix[:, :, 0]
        ) / (12 * dx)

    return derivative



def finite_dif_4th_order_periodic_for(matrix, axis, dx):
    """
    Calculate the derivative of a 3D matrix along a specified axis using a 4th order finite difference scheme
    with periodic boundary conditions. Central difference is applied for the boundary points.

    Parameters:
    - matrix: np.array, the 3D input matrix (shape: Nx * Ny * Nz).
    - axis: int, the axis along which to compute the derivative (0, 1, or 2).
    - dx: float, the spacing between points along the given axis.

    Returns:
    - derivative: np.array, the 3D matrix of derivatives with the same shape as input.
    """
    # Initialize the derivative array
    derivative = np.zeros_like(matrix)
    Nx, Ny, Nz = matrix.shape

    # 4th order central finite difference coefficients
    # def compute_derivative(i, j, k, axis):
    #     if axis == 0:  # x-axis
    #         return (matrix[i-2, j, k] - 8 * matrix[i-1, j, k] + 8 * matrix[i+1, j, k] - matrix[i+2, j, k]) / (12 * dx)
    #     elif axis == 1:  # y-axis
    #         return (matrix[i, j-2, k] - 8 * matrix[i, j-1, k] + 8 * matrix[i, j+1, k] - matrix[i, j+2, k]) / (12 * dx)
    #     elif axis == 2:  # z-axis
    #         return (matrix[i, j, k-2] - 8 * matrix[i, j, k-1] + 8 * matrix[i, j, k+1] - matrix[i, j, k+2]) / (12 * dx)
    
    def compute_derivative(i, j, k, axis):
        if axis == 0:  # x-axis
            fn = (i+1)*(i+1< Nx) + (0)*(i+1==Nx)
            sn = (i+2)*(i+2< Nx) + (0)*(i+2==Nx) + (1)*(i+2>Nx)
            fp = (i-1)*(i-1>-1) + (-1)*(i-1==-1)
            sp = (i-2)*(i-2>-1) + (-1)*(i-2==-1) + (-2)*(i-2<-1)
            return (matrix[sp, j, k] - 8 * matrix[fp, j, k] + 8 * matrix[fn, j, k] - matrix[sn, j, k]) / (12 * dx)
        elif axis == 1:  # y-axis
            fn = (j+1)*(j+1< Ny) + (0)*(j+1==Ny)
            sn = (j+2)*(j+2< Ny) + (0)*(j+2==Ny) + (1)*(j+2>Ny)
            fp = (j-1)*(j-1>-1) + (-1)*(j-1==-1)
            sp = (j-2)*(j-2>-1) + (-1)*(j-2==-1) + (-2)*(j-2<-1)
            return (matrix[i, sp, k] - 8 * matrix[i, fp, k] + 8 * matrix[i, fn, k] - matrix[i, sn, k]) / (12 * dx)
        elif axis == 2:  # z-axis
            fn = (k+1)*(k+1< Nz) + (0)*(k+1==Nz)
            sn = (k+2)*(k+2< Nz) + (0)*(k+2==Nz) + (1)*(k+2>Nz)
            fp = (k-1)*(k-1>-1) + (-1)*(k-1==-1)
            sp = (k-2)*(k-2>-1) + (-1)*(k-2==-1) + (-2)*(k-2<-1)
            return (matrix[i, j, sp] - 8 * matrix[i, j, fp] + 8 * matrix[i, j, fn] - matrix[i, j, sn]) / (12 * dx)    

    # Loop over each element in the matrix (excluding boundaries)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                derivative[i, j, k] = compute_derivative(i, j, k, axis)
    return derivative


