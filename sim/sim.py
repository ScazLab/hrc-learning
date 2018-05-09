# sim.py
# Michael Van der Linden, cs490
# Python 3

# This module simulates repeated interactions with the robot using various exploit-explore strategies and logs the results.
# It runs 12 sets of 250 iterations of 250 trials. 1 set for each variation of each strategy.
# The quality of the robot's performance for each trial in each iteration is logged in a csv file. Each row is an iteration.
# One csv file is generated for each set.

# import sys
import random
import pickle

NSTATES = 71	# number of states
NACTIONS = 8	# number of actions
STARTSTATE = 0	# starting state, if we even care
ENDSTATE = 70	# end state, assuming are using a predefined end
GAMMA = .3		# Q learning discount factor for future rewards

# returns the index of the max item in list. Randomly selects from max-indices if max appears multiple times
def rand_idx_max(lst):
	m = max(lst)
	l = [i for i, v in enumerate(lst) if v == m]
	return random.choice(l)

# Explore every possibility once until none are unexplored, then exploit
def explorefirst(options, trial, var):
	unexplored = [i for i, v in enumerate(options) if v == 0]
	nonneg     = [i for i, v in enumerate(options) if v >= 0]
	if unexplored:
		if var == 'A':
			return random.choice(nonneg) # return any random nonneg move
		else:							# var B or C. return one of unexplored
			return random.choice(unexplored)
	return rand_idx_max(options) # Exploit: pick (one of) the best

# chooses an action using epsilon-greedy strategy, where e is some small probability of exploring instead of exploiting
def epsilongreedy(options, trial, var):
	e = .2
	r = random.random()
	if r > e:
		return rand_idx_max(options) # Exploit: pick (one of) the best
	else:
		unexplored = [i for i, v in enumerate(options) if v == 0]
		nonneg     = [i for i, v in enumerate(options) if v >= 0]
		if var == 'A':
			return random.choice(nonneg) # Explore: pick randomly
		if unexplored:
			return random.choice(unexplored)
		if var == 'B':
			return random.choice(nonneg) # Explore: pick randomly
		return rand_idx_max(options) # C

# In early trials, explore all the time, gradually diminishing to 10% explore.
def epsilondecreasing(options, trial, var):
	e = max(.1, (15-trial)/15) # 1, .9, .8, .7, .6, .5, .4, .3, .2, .1, .1, .1, .1, ...
	r = random.random()
	if r > e:
		return rand_idx_max(options) # Exploit: pick (one of) the best
	else:
		unexplored = [i for i, v in enumerate(options) if v == 0]
		nonneg     = [i for i, v in enumerate(options) if v >= 0]
		if var == 'A':
			return random.choice(nonneg) # Explore: pick randomly
		if unexplored:
			return random.choice(unexplored)
		if var == 'B':
			return random.choice(nonneg) # Explore: pick randomly
		return rand_idx_max(options) # C

# Likelihood of exploiting best option is proportional to quality of best option found.
# Quirky because any option with value > 10 will always be exploited. Very arbitrary threshold. (issues with heavy backprop reinforcing a medium-grade option and shutting out a better one). Happened to perform well for this setup, but not generalizable
def epsilonproportional(options, trial, var):
	m = max(options)
	e = (10-m)/10 # chance of exploring
	r = random.random()
	if r > e:
		return rand_idx_max(options) # Exploit: pick (one of) the best
	else:
		unexplored = [i for i, v in enumerate(options) if v == 0]
		nonneg     = [i for i, v in enumerate(options) if v >= 0]
		if var == 'A':
			return random.choice(nonneg) # Explore: pick randomly
		if unexplored:
			return random.choice(unexplored)
		if var == 'B':
			return random.choice(nonneg) # Explore: pick randomly
		return rand_idx_max(options) # C


# run 250 iterations of 250 trials each, with a certain epsilon function and a certain variation, logging the total reward for each trial in each iteration
def simulate(R, T, A, strat, var):
	f = open(strat.__name__ + var + ".csv", 'w')
	for iteration in range(250):
		Q = [[0 for j in range(NACTIONS)] for i in range(NSTATES)] # Initialize matrix Q to 0s
		for trial in range(250):
			current_state = STARTSTATE					# Each episode starts from starting state
			total_reward = 0
			while (current_state != ENDSTATE):	# The robot keeps acting until it reaches end state. ! Maybe we don't need predefined end states
				action = strat(Q[current_state], trial, var)	# biased choice favoring good options with some exploration
				reward = R[current_state][action]
				total_reward = total_reward + R[current_state][action] # -1 reward for each red button move. Matches T-matrix
				next_state = T[current_state][action]
				if next_state != -1:  # green button. Moving on to next state
					Q[current_state][action] = R[current_state][action] + GAMMA * max(Q[next_state])
					current_state = next_state # if robot tries an action that doesn't lead anywhere, state does not change (self loop)
				else:   # red button. Staying within same state
					Q[current_state][action] = -20

			# end of trial, log reward
			f.write(str(total_reward))
			f.write(',')

		# end of iteration
		f.write('\n')
		# store the final Q matrix and print best path, just for the last iteration so we're not overwhelmed with output
		if iteration == 249:
			actions, reward = best_actions(R, T, A, Q)
			print('Best action sequence (' + strat.__name__ + ', variant ' + var + '):')
			print(actions)
			print("Total reward: %d" % reward)
			with open(strat.__name__ + var + '_final_Q.pickle', 'wb') as handle:
				pickle.dump(Q, handle, protocol=2)
	
	# end of all 250 iterations
	f.close()


# returns an ordered list of actions charting the best path through Q, and the total reward of those actions
def best_actions(R, T, A, Q):
	current_state = STARTSTATE
	actions = []
	total_reward = 0
	while(current_state != ENDSTATE):
		action = rand_idx_max(Q[current_state]) # index of max value in Q[current_state]
		actions.append(A[action])
		total_reward = total_reward + R[current_state][action]	# add latest reward to total reward
		current_state = T[current_state][action]
		if current_state < 0:
			print("Warning: Q matrix picked invalid action!")
			break
	return actions, total_reward

# Ordered list of actions
A = ["grab dowel",
	 "grab front bracket",
	 "grab back bracket",
	 "grab seat",
	 "grab top bracket",
	 "grab long dowel",
	 "grab back",
	 "hold"]

# Basic Reward Matrix (hard rules onle): (s,a) --> r
# R = [[10, 10, 10, -1, 10, -1, -1, -1],
# 	 [-1, 10, 10, -1, 10, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, 10, 10, -1, -1, -1, -1, -1],
# 	 [10, 10, 10, -1, -1, -1, -1, -1],
# 	 [-1, 10, 10, -1, -1, -1, -1, -1],
# 	 [-1, 10, 10, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, 10, -1, -1, -1, -1, -1],
# 	 [10, 10, -1, -1, -1, -1, -1, -1],
# 	 [10, 10, 10, -1, -1, -1, -1, -1],
# 	 [-1, -1, 10, -1, -1, -1, -1, -1],
# 	 [-1, 10, -1, -1, -1, -1, -1, -1],
# 	 [-1, 10, 10, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, 10, -1, -1, -1, -1, -1],
# 	 [10, 10, -1, -1, -1, -1, -1, -1],
# 	 [-1, -1, 10, -1, -1, -1, -1, -1],
# 	 [-1, 10, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [-1, -1, -1, 10, -1, -1, -1, -1],
# 	 [-1, -1, -1, -1, -1, -1, -1, 10],
# 	 [10, -1, -1, -1, 10, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, 10, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [-1, -1, -1, -1, 10, -1, -1, -1],
# 	 [-1, -1, -1, -1, -1, 10, 10, -1],
# 	 [-1, -1, -1, -1, -1, -1, 10, -1],
# 	 [-1, -1, -1, -1, -1, 10, -1, -1],
# 	 [-1, -1, -1, -1, -1, -1, -1, 10],
# 	 [10, 10, 10, -1, -1, -1, -1, -1],
# 	 [-1, 10, 10, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, 10, 10, -1, -1, -1, -1, -1],
# 	 [10, 10, 10, -1, -1, -1, -1, -1],
# 	 [-1, 10, 10, -1, -1, -1, -1, -1],
# 	 [-1, 10, 10, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, 10, -1, -1, -1, -1, -1],
# 	 [10, 10, -1, -1, -1, -1, -1, -1],
# 	 [10, 10, 10, -1, -1, -1, -1, -1],
# 	 [-1, -1, 10, -1, -1, -1, -1, -1],
# 	 [-1, 10, -1, -1, -1, -1, -1, -1],
# 	 [-1, 10, 10, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, 10, -1, -1, -1, -1, -1],
# 	 [10, 10, -1, -1, -1, -1, -1, -1],
# 	 [-1, -1, 10, -1, -1, -1, -1, -1],
# 	 [-1, 10, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [-1, -1, -1, 10, -1, -1, -1, -1],
# 	 [-1, -1, -1, -1, -1, -1, -1, 10],
# 	 [-1, -1, -1, -1, 10, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [10, -1, -1, -1, 10, -1, -1, -1],
# 	 [10, -1, -1, -1, -1, -1, -1, -1],
# 	 [-1, -1, -1, -1, 10, -1, -1, -1],
# 	 [-1, -1, -1, -1, -1, 10, 10, -1],
# 	 [-1, -1, -1, -1, -1, -1, 10, -1],
# 	 [-1, -1, -1, -1, -1, 10, -1, -1],
# 	 [-1, -1, -1, -1, -1, -1, -1, 10],
# 	 [-1, -1, -1, -1, -1, -1, -1, -1]]

# Preferential Reward Matrix (rules and preferences): (s,a) --> r
R = [[ 1,  3,  3, -1, 10, -1, -1, -1],
	 [-1,  1,  3, -1, 10, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [ 3,  5, 10, -1, -1, -1, -1, -1],
	 [ 3,  5, 10, -1, -1, -1, -1, -1],
	 [-1,  5, 10, -1, -1, -1, -1, -1],
	 [-1,  5, 10, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [ 3, -1, 10, -1, -1, -1, -1, -1],
	 [ 3, 10, -1, -1, -1, -1, -1, -1],
	 [ 3,  5, 10, -1, -1, -1, -1, -1],
	 [-1, -1, 10, -1, -1, -1, -1, -1],
	 [-1, 10, -1, -1, -1, -1, -1, -1],
	 [-1,  5, 10, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [ 2, -1, 10, -1, -1, -1, -1, -1],
	 [ 3, 10, -1, -1, -1, -1, -1, -1],
	 [-1, -1, 10, -1, -1, -1, -1, -1],
	 [-1, 10, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, 10, -1, -1, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 10],
	 [ 3, -1, -1, -1, 10, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [ 3, -1, -1, -1, 10, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, -1, 10, -1, -1, -1],
	 [-1, -1, -1, -1, -1,  5, 10, -1],
	 [-1, -1, -1, -1, -1, -1, 10, -1],
	 [-1, -1, -1, -1, -1, 10, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 10],
	 [ 3,  5, 10, -1, -1, -1, -1, -1],
	 [-1,  3, 10, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [ 3,  5, 10, -1, -1, -1, -1, -1],
	 [ 3,  5, 10, -1, -1, -1, -1, -1],
	 [-1,  3, 10, -1, -1, -1, -1, -1],
	 [-1,  3, 10, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [ 3, -1, 10, -1, -1, -1, -1, -1],
	 [ 3, 10, -1, -1, -1, -1, -1, -1],
	 [ 3,  5, 10, -1, -1, -1, -1, -1],
	 [-1, -1, 10, -1, -1, -1, -1, -1],
	 [-1, 10, -1, -1, -1, -1, -1, -1],
	 [-1,  3, 10, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [ 3, -1, 10, -1, -1, -1, -1, -1],
	 [ 3, 10, -1, -1, -1, -1, -1, -1],
	 [-1, -1, 10, -1, -1, -1, -1, -1],
	 [-1, 10, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, 10, -1, -1, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 10],
	 [-1, -1, -1, -1, 10, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [ 3, -1, -1, -1, 10, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, -1, 10, -1, -1, -1],
	 [-1, -1, -1, -1, -1,  3, 10, -1],
	 [-1, -1, -1, -1, -1, -1, 10, -1],
	 [-1, -1, -1, -1, -1, 10, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 10],
	 [-1, -1, -1, -1, -1, -1, -1, -1]]


# Transition Matrix: (s,a) --> s'
T = [[ 1,  2,  3, -1, 27, -1, -1, -1],
	 [-1,  4,  5, -1, 28, -1, -1, -1],
	 [ 4, -1, -1, -1, -1, -1, -1, -1],
	 [ 5, -1, -1, -1, -1, -1, -1, -1],
	 [ 6,  8, 10, -1, -1, -1, -1, -1],
	 [ 7, 10,  9, -1, -1, -1, -1, -1],
	 [-1, 11, 13, -1, -1, -1, -1, -1],
	 [-1, 13, 12, -1, -1, -1, -1, -1],
	 [11, -1, -1, -1, -1, -1, -1, -1],
	 [12, -1, -1, -1, -1, -1, -1, -1],
	 [13, -1, -1, -1, -1, -1, -1, -1],
	 [14, -1, 17, -1, -1, -1, -1, -1],
	 [15, 18, -1, -1, -1, -1, -1, -1],
	 [16, 17, 18, -1, -1, -1, -1, -1],
	 [-1, -1, 19, -1, -1, -1, -1, -1],
	 [-1, 20, -1, -1, -1, -1, -1, -1],
	 [-1, 19, 20, -1, -1, -1, -1, -1],
	 [19, -1, -1, -1, -1, -1, -1, -1],
	 [20, -1, -1, -1, -1, -1, -1, -1],
	 [21, -1, 23, -1, -1, -1, -1, -1],
	 [22, 23, -1, -1, -1, -1, -1, -1],
	 [-1, -1, 24, -1, -1, -1, -1, -1],
	 [-1, 24, -1, -1, -1, -1, -1, -1],
	 [24, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, 25, -1, -1, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 26],
	 [61, -1, -1, -1, 62, -1, -1, -1],
	 [28, -1, -1, -1, -1, -1, -1, -1],
	 [30, -1, -1, -1, 29, -1, -1, -1],
	 [31, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, -1, 31, -1, -1, -1],
	 [-1, -1, -1, -1, -1, 32, 33, -1],
	 [-1, -1, -1, -1, -1, -1, 34, -1],
	 [-1, -1, -1, -1, -1, 34, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 35],
	 [36, 37, 38, -1, -1, -1, -1, -1],
	 [-1, 39, 40, -1, -1, -1, -1, -1],
	 [39, -1, -1, -1, -1, -1, -1, -1],
	 [40, -1, -1, -1, -1, -1, -1, -1],
	 [41, 43, 45, -1, -1, -1, -1, -1],
	 [42, 45, 44, -1, -1, -1, -1, -1],
	 [-1, 46, 48, -1, -1, -1, -1, -1],
	 [-1, 48, 47, -1, -1, -1, -1, -1],
	 [46, -1, -1, -1, -1, -1, -1, -1],
	 [47, -1, -1, -1, -1, -1, -1, -1],
	 [48, -1, -1, -1, -1, -1, -1, -1],
	 [49, -1, 52, -1, -1, -1, -1, -1],
	 [50, 53, -1, -1, -1, -1, -1, -1],
	 [51, 52, 53, -1, -1, -1, -1, -1],
	 [-1, -1, 54, -1, -1, -1, -1, -1],
	 [-1, 55, -1, -1, -1, -1, -1, -1],
	 [-1, 54, 55, -1, -1, -1, -1, -1],
	 [54, -1, -1, -1, -1, -1, -1, -1],
	 [55, -1, -1, -1, -1, -1, -1, -1],
	 [56, -1, 58, -1, -1, -1, -1, -1],
	 [57, 58, -1, -1, -1, -1, -1, -1],
	 [-1, -1, 59, -1, -1, -1, -1, -1],
	 [-1, 59, -1, -1, -1, -1, -1, -1],
	 [59, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, 60, -1, -1, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 70],
	 [-1, -1, -1, -1, 63, -1, -1, -1],
	 [63, -1, -1, -1, -1, -1, -1, -1],
	 [65, -1, -1, -1, 64, -1, -1, -1],
	 [66, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, -1, 66, -1, -1, -1],
	 [-1, -1, -1, -1, -1, 67, 68, -1],
	 [-1, -1, -1, -1, -1, -1, 69, -1],
	 [-1, -1, -1, -1, -1, 69, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 70],
	 [-1, -1, -1, -1, -1, -1, -1, -1]]


# Run simulations
simulate(R, T, A, explorefirst, 'A')
simulate(R, T, A, explorefirst, 'C')
simulate(R, T, A, epsilongreedy, 'A')
simulate(R, T, A, epsilongreedy, 'B')
simulate(R, T, A, epsilongreedy, 'C')
simulate(R, T, A, epsilondecreasing, 'A')
simulate(R, T, A, epsilondecreasing, 'B')
simulate(R, T, A, epsilondecreasing, 'C')
simulate(R, T, A, epsilonproportional, 'A')
simulate(R, T, A, epsilonproportional, 'B')
simulate(R, T, A, epsilonproportional, 'C')