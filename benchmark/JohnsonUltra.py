# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:24:36 2021

@author: pheno

Customized Johnson's func from networkx implementation
    Recovers acutal distance from dijkstra results
"""

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function, _bellman_ford, _dijkstra

def johnsonU(G, weight="weight"):
    '''
    Revise from networkx func to also return distances
    '''
    if not nx.is_weighted(G, weight=weight):
        raise nx.NetworkXError("Graph is not weighted.")

    dist = {v: 0 for v in G}
    pred = {v: [] for v in G}
    weight = _weight_function(G, weight)

    # Calculate distance of shortest paths
    dist_bellman = _bellman_ford(G, list(G), weight, pred=pred, dist=dist)

    # Update the weight function to take into account the Bellman--Ford
    # relaxation distances.
    def new_weight(u, v, d):
        return weight(u, v, d) + dist_bellman[u] - dist_bellman[v]

    # def dist_path(v):
    #     paths = {v: [v]}
    #     _dijkstra(G, v, new_weight, paths=paths)
    #     return paths

    def dist_path_all(v):
        paths = {v: [v]}
        dist_v_to_all =_dijkstra(G, v, new_weight, paths=paths)
        return paths, dist_v_to_all
    
    results_path = {}
    results_dist = {}
    # for v in G:
    #     paths = {v: [v]}
    #     dist_v_to_all =_dijkstra(G, v, new_weight, paths=paths)
    #     results_path[v] = paths
    #     results_dist[v] = dist_v_to_all
    
    for v in G:
        results_path[v], results_dist[v] = dist_path_all(v)
    
    # The distance in the original graph is then computed 
    # for each distance D(u , v), by adding h(v) − h(u) to 
    # the distance returned by Dijkstra's algorithm.
    actual_dist = {}
    for u in G:
        actual_dist[u] = {}
        for v in G:
            # self-loop
            if u == v:
                actual_dist[u][v] = results_dist[u][v]
            else:
                actual_dist[u][v] = results_dist[u][v] + dist_bellman[v] - dist_bellman[u]
                
    return results_path, actual_dist
    #return {v: dist_path(v) for v in G}
