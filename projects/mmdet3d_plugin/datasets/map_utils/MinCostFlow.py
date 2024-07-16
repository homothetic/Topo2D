"""
MinCostFow solver adapted for matching two set of contours. The implementation is based on google-ortools.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

from __future__ import print_function
import numpy as np
from ortools.graph import pywrapgraph
import time


def SolveMinCostFlow(adj_mat, cost_mat):
    """
        Solving an Assignment Problem with MinCostFlow"
    :param adj_mat: adjacency matrix with binary values indicating possible matchings between two sets
    :param cost_mat: cost matrix recording the matching cost of every possible pair of items from two sets
    :return:
    """

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Define the directed graph for the flow.

    cnt_1, cnt_2 = adj_mat.shape
    cnt_nonzero_row = int(np.sum(np.sum(adj_mat, axis=1) > 0))
    cnt_nonzero_col = int(np.sum(np.sum(adj_mat, axis=0) > 0))

    # prepare directed graph for the flow
    start_nodes = np.zeros(cnt_1, dtype=np.int).tolist() +\
                  np.repeat(np.array(range(1, cnt_1+1)), cnt_2).tolist() + \
                  [i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]
    end_nodes = [i for i in range(1, cnt_1+1)] + \
                np.repeat(np.array([i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]).reshape([1, -1]), cnt_1, axis=0).flatten().tolist() + \
                [cnt_1 + cnt_2 + 1 for i in range(cnt_2)]
    capacities = np.ones(cnt_1, dtype=np.int).tolist() + adj_mat.flatten().astype(np.int).tolist() + np.ones(cnt_2, dtype=np.int).tolist()
    costs = (np.zeros(cnt_1, dtype=np.int).tolist() + cost_mat.flatten().astype(np.int).tolist() + np.zeros(cnt_2, dtype=np.int).tolist())
    # Define an array of supplies at each node.
    supplies = [min(cnt_nonzero_row, cnt_nonzero_col)] + np.zeros(cnt_1 + cnt_2, dtype=np.int).tolist() + [-min(cnt_nonzero_row, cnt_nonzero_col)]
    # supplies = [min(cnt_1, cnt_2)] + np.zeros(cnt_1 + cnt_2, dtype=np.int).tolist() + [-min(cnt_1, cnt_2)]
    source = 0
    sink = cnt_1 + cnt_2 + 1

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])

    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    match_results = []
    # Find the minimum cost flow between node 0 and node 10.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        # print('Total cost = ', min_cost_flow.OptimalCost())
        # print()
        for arc in range(min_cost_flow.NumArcs()):

            # Can ignore arcs leading out of source or into sink.
            if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:

                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.

                if min_cost_flow.Flow(arc) > 0:
                    # print('set A item %d assigned to set B item %d.  Cost = %d' % (
                    #     min_cost_flow.Tail(arc)-1,
                    #     min_cost_flow.Head(arc)-cnt_1-1,
                    #     min_cost_flow.UnitCost(arc)))
                    match_results.append([min_cost_flow.Tail(arc)-1,
                                          min_cost_flow.Head(arc)-cnt_1-1,
                                          min_cost_flow.UnitCost(arc)])
    else:
        print('There was an issue with the min cost flow input.')

    return match_results


def main():
    """Solving an Assignment Problem with MinCostFlow"""

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Define the directed graph for the flow.

    start_nodes = [0, 0, 0, 0] + [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4] + [5, 6, 7, 8]
    end_nodes = [1, 2, 3, 4] + [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8] + [9, 9, 9, 9]
    capacities = [1, 1, 1, 1] + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]
    costs = ([0, 0, 0, 0] + [90, 76, 75, 70, 35, 85, 55, 65, 125, 95, 90, 105, 45, 110, 95, 115] + [0, 0, 0, 0])
    # Define an array of supplies at each node.
    supplies = [4, 0, 0, 0, 0, 0, 0, 0, 0, -4]
    source = 0
    sink = 9
    tasks = 4

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])

    # Add node supplies.

    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])
    # Find the minimum cost flow between node 0 and node 10.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        print('Total cost = ', min_cost_flow.OptimalCost())
        print()
        for arc in range(min_cost_flow.NumArcs()):

            # Can ignore arcs leading out of source or into sink.
            if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:

                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.

                if min_cost_flow.Flow(arc) > 0:
                    print('Worker %d assigned to task %d.  Cost = %d' % (
                        min_cost_flow.Tail(arc),
                        min_cost_flow.Head(arc),
                        min_cost_flow.UnitCost(arc)))
    else:
        print('There was an issue with the min cost flow input.')


if __name__ == '__main__':
    start_time = time.clock()
    main()
    print()
    print("Time =", time.clock() - start_time, "seconds")