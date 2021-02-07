# Author: Thunradee Tangsupakij
# Date: 5/5/2020
# Descriptions: This file contains functions that analyzing our PPR model, using Zipf distribution step lengths,
# by comparing with the PPR model that uses Geometric distribution step lengths.

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import analysis_functions as af

random.seed(591)

def convergence_plot(G, walker_num, iteration_num):
    '''
    Plots convergence rates of Zipf model and Geometric model
    '''
    num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    geom_errs_df_alphas = pd.DataFrame()
    zipf_errs_df_alphas = pd.DataFrame()

    # do for different parameters
    for n in num:
        start_node = '1'

        # Geometric
        geom_errs = []
        alpha = round(n * 0.1, 2)

        # Zipf
        zipf_errs = []
        a = n + 1

        # Compute errors between iterations, append to lists, repeat the process for 'iteration_num' times
        for i in range(iteration_num):
            # Geometric
            geom_ppr_scores, geom_SAEs = af.geom_random_walk(G, start_node=start_node, walker_num=walker_num, alpha=alpha)
            geom_errs.append(geom_SAEs)

            # Zipf
            zipf_ppr_scores, zipf_SAEs = af.zipf_random_walk(G, start_node=start_node, walker_num=walker_num, a=a)
            zipf_errs.append(zipf_SAEs)

        # find average
        # Geom
        geom_errs_df = pd.DataFrame(data=geom_errs)
        geom_errs_mean = geom_errs_df.mean(axis=0)  # mean by column, return Series
        col_name = "alpha= " + str(alpha)
        geom_errs_df_alphas[col_name] = geom_errs_mean.values

        # Zipf
        zipf_errs_df = pd.DataFrame(data=zipf_errs)
        zipf_errs_mean = zipf_errs_df.mean(axis=0)  # mean by column, return Series
        col_name = "a= " + str(a)
        zipf_errs_df_alphas[col_name] = zipf_errs_mean.values

    # Plot
    # Geom
    lines = geom_errs_df_alphas.plot.line()
    plt.title("Convergence Plot of Geometric Distribution")
    plt.xlabel("Number of Random Walks")
    plt.ylabel("Sum of Absolute PR Score Differences")
    plt.savefig("Geometric_convergence.png")
    # plt.show()

    # Zipf
    lines = zipf_errs_df_alphas.plot.line()
    plt.title("Convergence Plot of Zipf Distribution")
    plt.xlabel("Number of Random Walks")
    plt.ylabel("Sum of Absolute PR Score Differences")
    plt.savefig("Zipf_convergence.png")
    plt.show()


def mean_abs_deviation(G, walker_num, iteration_num):
    '''
    Plots error mean absolute deviation
    '''
    num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    geom_mad_df_params = pd.DataFrame()
    zipf_mad_df_params = pd.DataFrame()

    # do for different parameters
    for n in num:
        start_node = '1'

        # Geometric
        geom_errs = []
        alpha = round(n * 0.1, 2)

        # Zipf
        zipf_errs = []
        a = n + 1

        # Compute errors between iterations, append to lists, repeat the process for 'iteration_num' times
        for i in range(iteration_num):
            # Geometric
            geom_ppr_scores, geom_SAEs = af.geom_random_walk(G, start_node=start_node, walker_num=walker_num, alpha=alpha)
            geom_errs.append(geom_SAEs)

            # Zipf
            zipf_ppr_scores, zipf_SAEs = af.zipf_random_walk(G, start_node=start_node, walker_num=walker_num, a=a)
            zipf_errs.append(zipf_SAEs)

        # mean absolute deviation
        # 1/n(sum(|err - its mean|))
        # Geom
        geom_errs_df = pd.DataFrame(data=geom_errs)
        geom_errs_dev = geom_errs_df - geom_errs_df.mean(axis=0)  # subtract column's mean from the column values
        geom_errs_abs_dev = geom_errs_dev.abs()  # make the value absolute
        geom_errs_mad = geom_errs_abs_dev.mean(axis=0)  # mean by column, return Series
        col_name = "alpha= " + str(alpha)
        geom_mad_df_params[col_name] = geom_errs_mad.values

        # Zipf
        zipf_errs_df = pd.DataFrame(data=zipf_errs)
        zipf_errs_dev = zipf_errs_df - zipf_errs_df.mean(axis=0)  # subtract column's mean from the column values
        zipf_errs_abs_dev = zipf_errs_dev.abs()  # make the value absolute
        zipf_errs_mad = zipf_errs_abs_dev.mean(axis=0)  # mean by column, return Series
        col_name = "a= " + str(a)
        zipf_mad_df_params[col_name] = zipf_errs_mad.values

    # Plot
    # Geom
    lines = geom_mad_df_params.plot.line()
    plt.title("Mean Absolute Deviation of Geometric Distribution")
    plt.xlabel("Number of Random Walks")
    plt.ylabel("Mean Absolute Deviation")
    plt.savefig("Geometric_mad.png")
    # plt.show()

    # Zipf
    lines = zipf_mad_df_params.plot.line()
    plt.title("Mean Absolute Deviation of Zipf Distribution")
    plt.xlabel("Number of Random Walks")
    plt.ylabel("Mean Absolute Deviation")
    plt.savefig("Zipf_mad.png")
    plt.show()


def num_walks_to_converge(G, walker_num, iteration_num):
    '''
    Plots number of random walks used for convergence
    '''
    num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    geom_walk_num_params = []
    zipf_walk_num_params = []

    # do for different parameters
    for n in num:
        start_node = '1'

        # Geometric
        geom_walk_nums = []
        alpha = round(n * 0.1, 2)

        # Zipf
        zipf_walk_nums = []
        a = n + 1

        # Compute number of random walks needed to converge, append to lists, repeat the process for 'iteration_num' times
        for i in range(iteration_num):
            _, _, geom_walk_num = af.geom_random_walk(G, start_node=start_node, walker_num=walker_num, alpha=alpha, threshold=True)
            geom_walk_nums.append(geom_walk_num)

            _, _, zipf_walk_num = af.zipf_random_walk(G, start_node=start_node, walker_num=walker_num, a=a, threshold=True)
            zipf_walk_nums.append(zipf_walk_num)

        # find mean
        geom_mean = sum(geom_walk_nums)/len(geom_walk_nums)
        zipf_mean = sum(zipf_walk_nums)/len(zipf_walk_nums)

        # append to list
        geom_walk_num_params.append(geom_mean)
        zipf_walk_num_params.append(zipf_mean)

    # plot
    # Geom
    index = [str(round((n*0.1), 2)) for n in num]
    geom_df = pd.DataFrame({'Alpha': index,
                            'Number of random walks': geom_walk_num_params})
    ax = geom_df.plot.bar(x='Alpha', y='Number of random walks', rot=0, legend=False)
    plt.title("Convergence By Number of Walks (Geometric Distribution)")
    plt.xlabel("Alpha")
    plt.ylabel("Number of Random Walks")
    plt.savefig("Geom_walk_num.png")
    plt.show()

    # Zipf
    index = [str(n+1) for n in num]
    zipf_df = pd.DataFrame({'A': index,
                            'Number of random walks': zipf_walk_num_params})
    ax = zipf_df.plot.bar(x='A', y='Number of random walks', rot=0, legend=False)
    plt.title("Convergence By Number of Walks (Zipf Distribution)")
    plt.xlabel("A")
    plt.ylabel("Number of Random Walks")
    plt.savefig("zipf_walk_num.png")
    plt.show()


def ranks_scores_difference(G, walker_num, iteration_num):
    '''
    Plots top 5 absolute ranks and scores difference between results using Geometric model and Zipf model
    '''
    start_node = '1'

    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alist = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    rank_dif_matrix = []
    score_dif_matrix = []

    # loop thru different Zipf parameters
    for a in alist:
        rank_dif_list = []
        score_dif_list = []
        # loop thru different Geometric parameters
        for alp in alpha:
            geom_ppr_scores, _, _ = af.geom_random_walk(G, start_node=start_node, walker_num=walker_num, alpha=alp, threshold=True)
            geom_rank = af.gen_rank(geom_ppr_scores)
            zipf_ppr_scores, _, _ = af.zipf_random_walk(G, start_node=start_node, walker_num=walker_num, a=a, threshold=True)
            zipf_rank = af.gen_rank(zipf_ppr_scores)

            top = 5

            # get the top 5 node accept the starting node
            # geom_nodes = list(geom_rank.keys())[1:top+1]

            # get the top 5 node
            geom_nodes = list(geom_rank.keys())[:top]

            # calculate absolute rank difference
            rank_dif = [abs(geom_rank[x] - zipf_rank[x]) for x in geom_nodes]

            # calculate absolute scores differance
            score_dif = [abs(geom_ppr_scores[x] - zipf_ppr_scores[x]) for x in geom_nodes]

            sum_rank_dif = sum(rank_dif)
            sum_score_dif = sum(score_dif)

            rank_dif_list.append(sum_rank_dif)
            score_dif_list.append(sum_score_dif)

        rank_dif_matrix.append(rank_dif_list)
        score_dif_matrix.append(score_dif_list)

        alpha_str = [str(x) for x in alpha]
        a_str = [str(x) for x in alist]

    # plot
    rank_dif_df = pd.DataFrame(rank_dif_matrix, columns=alpha_str, index=a_str)
    rank_dif_df.index.name = 'Zipf Distribution Parameter (A)'
    rank_dif_df.columns.name = 'Geometric distribution Parameter (Alpha)'
    plt.figure(figsize=(10, 10))
    plt.title("Rank Difference Matrix")
    sns.set(font_scale=0.8)  # for label size
    sns.heatmap(rank_dif_df, cmap="Blues", annot=True, annot_kws={"size": 8})  # font size
    plt.savefig("rank_dif.png")
    plt.show()

    score_dif_df = pd.DataFrame(score_dif_matrix, columns=alpha_str, index=a_str)
    score_dif_df.index.name = 'Zipf Distribution Parameter (A)'
    score_dif_df.columns.name = 'Geometric distribution Parameter (Alpha)'
    plt.figure(figsize=(10, 10))
    plt.title("PPR Score Difference Matrix")
    sns.set(font_scale=0.8)  # for label size
    sns.heatmap(score_dif_df, cmap="Blues", annot=True, annot_kws={"size": 8})  # font size
    plt.savefig("score_dif.png")
    plt.show()
