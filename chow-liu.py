'''
    @author: Emile Hreich
             emile.janhodithreich@epfl.ch

             Emile Charles
             emile.charles@epfl.ch

    @file: nn.py

    @date: 09/06/2022

    @brief: Homework 3 problem 2

'''
# ========================================================================
# Libraries

import numpy as N 
import networkx as nx
from collections import defaultdict


# ========================================================================
# Helper functions

def marginal_distribution(X, U):
    """
        Computes the marginal distribution of U.
    """
    values = defaultdict(float)
    s = 1. / len(X)
    for x in X:
        values[x[U]] += s
    return values



def joint_distribution(X, U, V):
    """
        Computes joint distribution
    """
    if U > V:
        U, V = V, U

    values = defaultdict(float)
    s = 1. / len(X)
    for x in X:
        values[(x[U], x[V])] += s

    return values



def mutual_information(X, u, v):
    """
        Computes the mutual information of two features

        params:
            X   :  data points.
            u, v:  the indices of the features to calculate the mutual information for.
    """
    if u > v:
        u, v = v, u

    marginal_u  = marginal_distribution(X, u)
    marginal_v  = marginal_distribution(X, v)
    marginal_uv = joint_distribution(X, u, v)

    I = 0.

    for x_u, p_x_u in marginal_u.iteritems():
        for x_v, p_x_v in marginal_v.iteritems():

            if (x_u, x_v) in marginal_uv:
                p_x_uv = marginal_uv[(x_u, x_v)]

                # based on the given formula in the assignment
                I += p_x_uv * (N.log(p_x_uv) - N.log(p_x_u) - N.log(p_x_v))
    return I

# ========================================================================
# Chow-Liu Algorithm implementation

def build_chow_liu_tree(X, n):
    """
    Build a Chow-Liu tree from the data, X. n is the number of features. The weight on each edge is
    the negative of the mutual information between those features. The tree is returned as a networkx
    object.
    """
    G = nx.Graph()
    for v in range(n):
        G.add_node(v)
        for u in range(v):
            G.add_edge(u, v, weight = -mutual_information(X, u, v))

    T = nx.minimum_spanning_tree(G)
    return T
            


if '__main__' == __name__:
    import doctest
    doctest.testmod()