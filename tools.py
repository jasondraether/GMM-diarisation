from itertools import chain, combinations, product
import numpy as np

# Helper functions for main body
def powerset(iterable):
    s = list(iterable)
    # Excludes empty set
    pset = list(chain.from_iterable(combinations(s,r) for r in range(1,len(s)+1)))
    # Turn into a list of lists (default list conversion has a bunch of tuples)
    pset_list = []
    for set in pset:
        subset = []
        for element in set:
            subset.append(element)
        pset_list.append(subset)
    return pset_list

def cartesian_product(iterable):
    c_product = list(product(*iterable))

    product_list = []
    for p in c_product:
        sublist = []
        for element in p:
            sublist.append(element)
        product_list.append(sublist)
    return product_list
