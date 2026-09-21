def velocity_name_FI(file_name):
    """
    Given a file name, return the corresponding velocity name in the format 'Velocity_xxxx'
    where xxxx is a four-digit number. If the file name does not match any of the predefined
    cases, return None.

    Parameters
    ----------
    file_name : str
        The file name to be converted.

    Returns
    -------
    str
        The corresponding velocity name.
    """
    if file_name == 'U_t0':
        return 'Velocity_0001'
    elif file_name == 'U_t1':
        return 'Velocity_0501'
    elif file_name == 'U_t2':
        return 'Velocity_1001'
    elif file_name == 'U_t3':
        return 'Velocity_1501'
    else:
        return None


def velocity_name_ChF(file_name):
   
    """
    Given a file name, return the corresponding velocity name in the format 'Velocity_xxxx'
    where xxxx is a four-digit number. If the file name does not match any of the predefined
    cases, return None.

    Parameters
    ----------
    file_name : str
        The file name to be converted.

    Returns
    -------
    str
        The corresponding velocity name.
    """

    if file_name == 'U_t1':
        return 'Velocity_0001'
    elif file_name == 'U_t1301':
        return 'Velocity_1301'
    elif file_name == 'U_t2601':
        return 'Velocity_2601'
    elif file_name == 'U_t3901':
        return 'Velocity_3901'
    else:
        return None


