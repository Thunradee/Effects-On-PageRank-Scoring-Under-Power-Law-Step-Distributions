# Author: Thunradee Tangsupakij
# Date: 5/5/2020
# Descriptions: This file contains functions that used for analyzing our PPR model.

import numpy as np
import networkx as nx
import random

random.seed(591)

def weighted_choice(objects, weights):
    '''
    Select an object by weights
    '''
    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    # standardization:
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random.uniform(0, 1)
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]


def geom_random_walk(G, start_node, walker_num, alpha, tol=1.0e-6, threshold=False):
    '''
    Simulate random walks using Geometric distribution step lengths
    '''
    # number of nodes in the network
    N = len(G)

    # create a dictionary of PR scores by nodes
    nodes = list(map(str, [*range(0, len(G), 1)]))
    walkers = [0] * len(G)
    scores = dict(zip(nodes, walkers))

    # scores of previous iteration
    prev_scores = scores.copy()

    # score of 100 previous iteration
    last_100_scores = scores.copy()

    # sum absolute errors
    SAEs = []

    # start simulation
    for w in range(1, walker_num+1, 1):
        cur_node = start_node
        while True:
            # select the next move
            move = weighted_choice(['walk', 'terminate'], [1 - alpha, alpha])
            # the next move is walking
            if move == 'walk':
                adj_list = list(G[cur_node]._atlas.keys())
                if len(adj_list) > 0:
                    next_node = random.choice(adj_list)
                    cur_node = next_node
            # the next move is jumping back to the initial node
            else:
                break

        # increment number of walker of the node that the random walk landed on by 1
        scores[cur_node] = scores[cur_node] + 1

        # finding the errors between iteration
        cur_scores = scores.copy()
        s = float(sum(cur_scores.values()))
        cur_scores = dict((k, v / s) for k, v in cur_scores.items())
        SAE = sum_abs_error(prev_scores, cur_scores)
        SAEs.append(SAE)
        prev_scores = cur_scores.copy()

        # use threshold to terminate the simulation
        if threshold == True:
            # calculate the error every 100 iterations
            if (w % 100) == 0:
                cur_scores = scores.copy()
                s = float(sum(cur_scores.values()))
                cur_scores = dict((k, v / s) for k, v in cur_scores.items())
                err_last_100 = sum_abs_error(last_100_scores, cur_scores)
                last_100_scores = cur_scores.copy()
                # terminate if the error less than the threshold
                if err_last_100 < (N*tol):
                    return cur_scores, SAEs, w

    # terminate after the max iteration
    # calculate scores
    s = float(sum(scores.values()))
    scores = dict((k, v / s) for k, v in scores.items())
    return scores, SAEs, walker_num


def zipf_random_walk(G, start_node, walker_num, a, tol=1.0e-6, threshold=False):
    '''
    Simulate random walks using zipf distribution step lengths
    '''
    # number of nodes in the network
    N = len(G)

    # create a dictionary of PR scores by nodes
    nodes = list(map(str, [*range(0, len(G), 1)]))
    walkers = [0] * len(G)
    scores = dict(zip(nodes, walkers))

    # scores of previous iteration
    prev_scores = scores.copy()

    # score of 100 previous iteration
    last_100_scores = scores.copy()

    # sum absolute errors
    SAEs = []

    # start simulation
    for w in range(1, walker_num+1, 1):
        # generate a step length
        steps = np.random.zipf(a=a)
        cur_node = start_node

        # subtract 1 to shift the range because the function only generate number from 1 to n
        for s in range(steps - 1):
            adj_list = list(G[cur_node]._atlas.keys())
            if len(adj_list) > 0:
                next_node = random.choice(adj_list)
                cur_node = next_node

        # increment number of walker of the node that the random walk landed on by 1
        scores[cur_node] = scores[cur_node] + 1

        # finding the errors between iteration
        cur_scores = scores.copy()
        s = float(sum(cur_scores.values()))
        cur_scores = dict((k, v / s) for k, v in cur_scores.items())
        SAE = sum_abs_error(prev_scores, cur_scores)
        SAEs.append(SAE)
        prev_scores = cur_scores.copy()

        # calculate the error every 100 iterations
        if threshold == True:
            # calculate the error every 100 iterations
            if (w % 100) == 0:
                cur_scores = scores.copy()
                s = float(sum(cur_scores.values()))
                cur_scores = dict((k, v / s) for k, v in cur_scores.items())
                err_last_100 = sum_abs_error(last_100_scores, cur_scores)
                last_100_scores = cur_scores.copy()
                # terminate if the error less than the threshold
                if err_last_100 < (N * tol):
                    return cur_scores, SAEs, w

    # terminate after the max iteration
    # calculate scores
    s = float(sum(scores.values()))
    scores = dict((k, v / s) for k, v in scores.items())
    return scores, SAEs, walker_num


def sum_abs_error(ranks1: dict, ranks2: dict):
    '''
    Calculate sum absolute errors
    '''
    SAE = 0
    for key, value in ranks1.items():
        SAE += abs(value - ranks2[key])

    return SAE


def gen_rank(scores: dict):
    '''
    Generate ranks from PR scores
    '''
    scores_sorted = list(scores.items())
    scores_sorted.sort(key=lambda x: x[1], reverse=True)
    ranks = {}

    tie = 0
    for i in range(len(scores_sorted)):
        if i == 0:
            key, val = scores_sorted[i]
            ranks[key] = i
        else:
            p_key, p_val = scores_sorted[i - 1]  # previous node
            key, val = scores_sorted[i]  # current node

            if val < p_val:
                ranks[key] = i
                tie = 0
            else:  # tie
                ranks[key] = i - 1 - tie
                tie += 1

    return ranks
