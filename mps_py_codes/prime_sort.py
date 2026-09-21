from sympy import factorint

def prime_sort(u_shape):
    
    """
    This function takes a 3-tuple u_shape and returns two outputs:
    1. A tuple of the prime factors of the elements of u_shape, sorted in descending order
    2. A 3-tuple of the powers of the prime factors of the elements of u_shape
    3. A tuple of the prime factors of the elements of u_shape, sorted according to a custom rule: 
       the highest prime of each element of u_shape is picked first, then the next highest, etc.
       
    The custom sorting rule is used to optimize the reshaping and transposing of the matrix later on.
    """
    
    # Get prime factors and their powers
    x, y, z = u_shape
    factors_x = factorint(x)
    factors_y = factorint(y)
    factors_z = factorint(z)
    
    # Convert factors to a list of primes, repeating according to their powers
    primes_x = [p for p in factors_x for _ in range(factors_x[p])]
    primes_y = [p for p in factors_y for _ in range(factors_y[p])]
    primes_z = [p for p in factors_z for _ in range(factors_z[p])]
    
    # Sort primes in descending order within each factor list
    primes_x.sort(reverse=True)
    primes_y.sort(reverse=True)
    primes_z.sort(reverse=True)
    axis_powers = [len(primes_x), len(primes_y), len(primes_z)]

    # First output: Concatenate primes in the order of x, y, z
    ordered_primes = tuple(primes_x + primes_y + primes_z)
    
    # Initialize result tuple for second output
    result = []
    
    while primes_x or primes_y or primes_z:
        # Get highest primes from x, y, z
        if primes_x:
            result.append(primes_x.pop(0))
        if primes_y:
            result.append(primes_y.pop(0))
        if primes_z:
            result.append(primes_z.pop(0))
    
    # Second output: Custom sorted primes
    transpose_primes = tuple(result)
    
    return ordered_primes, axis_powers,transpose_primes

