# Author: Thunradee Tangsupakij
# Date: 5/5/2020

from model_analysis import *

random.seed(591)

# read data
fb = nx.read_edgelist("facebook_combined 2.txt")
krt = nx.read_edgelist("karate.edgelist")
phy = nx.read_edgelist("physicians.txt")

walker_num = 100
it_num = 100

# Model Analysis
# plot convergence rate
convergence_plot(fb, walker_num=walker_num, iteration_num=it_num)

# plot errors mean absolute deviation
mean_abs_deviation(fb, walker_num=walker_num, iteration_num=it_num)

# plot number of random walks needed to converge
num_walks_to_converge(fb, walker_num=30000, iteration_num=it_num)

# plot ranks absolute difference and scores absolute difference
# Facebook network
ranks_scores_difference(fb, walker_num=30000, iteration_num=it_num)

# Karate network
ranks_scores_difference(krt, walker_num=30000, iteration_num=it_num)

# Physicians network
ranks_scores_difference(phy, walker_num=30000, iteration_num=it_num)