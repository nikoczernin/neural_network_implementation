
def get_multidimensional_combinations(dims, all_perms=None, current_perm=None):
    # Functino that returns all possible combinations of parameters passed as lists
    # Example:
    # get_multidimensional_combinations([["a", "b"], [1, 2]])
    # returns
    # [['a', 1], ['a', 2], ['b', 1], ['b', 2]]
    # dims: list of lists of values
    # other parameters are for recursion, don't touch!!
    if all_perms is None: all_perms = []
    if current_perm is None: current_perm = []
    # iterate through the first dimension at hand
    for value in dims[0]:
        if len(dims)>1:
            get_multidimensional_combinations(dims[1:], all_perms, current_perm + [value])
        else: # only executed at last dimension
            all_perms.append(current_perm + [value])
    return all_perms
