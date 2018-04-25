# qlearn.py
# Michael Van der Linden, cs490
# Python 3

import sys
import random
import copy

NSTATES = 71	# number of states
NACTIONS = 8	# number of actions
STARTSTATE = 0	# starting state, if we even care
ENDSTATE = 70	# end state, assuming are using a predefined end
GAMMA = .9		# Q learning discount factor for future rewards
BLOCKSIZE = 10	# how many iterations to do in a batch before checking convergence
STREAKLEN = 5	# how many succesful trials in a row until Q matrix is good enough

# returns the index of the max item in list. Randomly selects from max-indices if max appears multiple times
def rand_idx_max(lst):
	m = max(lst)
	l = [i for i, v in enumerate(lst) if v == m]
	return random.choice(l)

# nicely prints a matrix
def printmat(Q):
	for row in Q:
		print(row)

# returns True if every cell in Qa is within a certain precision of its counterpart in Qb
def converged(Qa, Qb):
	if Qa == None or Qb == None:
		return False
	for i in range(NSTATES):
		for j in range(NACTIONS):
			if abs(Qa[i][j] - Qb[i][j]) > 1e-3: # this doesn't make a difference for some reason
				return False
	return True

# returns matrix holding the difference at each cell between Qa and Qb
def compmat(Qa, Qb):
	if Qa == None or Qb == None:
		return [["N/A"]]
	return [[abs(Qa[i][j] - Qb[i][j]) for j in range(NACTIONS)] for i in range(NSTATES)]

# chooses an action using epsilon-greedy strategy, where e is some small probability of exploring instead of exploiting
def exploitexplore(options, e):
	r = random.random()
	if r > e:
		return rand_idx_max(options) # Exploit: pick (one of) the best
	else:
		return random.randrange(len(options)) # Explore: pick randomly

def goodrun(total_reward):
	return total_reward == 170 # good run does 17 good actions with no detours or mistakes

# Runs trials one by one until the max path through Q was optimal after the last 5 trials
# ! Right now all the work is being done in the correction phase because the robot keeps trying at some current state without moving on until it gets a good move. Converges in 1 trial
def goodenoughQ(R, T):
	print("Starting learning iteration...")
	Q = [[-1 for j in range(NACTIONS)] for i in range(NSTATES)] # Initialize matrix Q to 0s
	streak = 0		# number of successful trials in a row
	trial = 0
	while streak < STREAKLEN:
		# run trials one at a time - no batching because we want minimal trials
		eps = max(0.1, 1/(trial+1))
		# current_state = random.randrange(NSTATES)	# Each episode starts from a random state
		current_state = STARTSTATE					# Each episode starts from starting state
		while (current_state != ENDSTATE):	# The robot keeps acting until it reaches end state. ! Maybe we don't need predefined end states
			action = exploitexplore(Q[current_state], eps)	# biased choice favoring good options with some exploration
			# action = random.randrange(NACTIONS)		# chooses next action randomly
			# action = rand_idx_max(Q[current_state])	# chooses best next action (better than random). But doesn't explore, so might miss better possibilities
			next_state = T[current_state][action]
			Q[current_state][action] = R[current_state][action] + GAMMA * (max(Q[next_state]) if next_state != -1 else -1)
			if next_state != -1:  # green button 
				current_state = next_state # if robot tries an action that doesn't lead anywhere, state does not change (self loop)
			else:   # red button
				print("self-looping")
		# end of trial
		trial = trial + 1
		total_reward = check_seq(Q, T, R)
		print(best_seq(Q, T, A, R)[0])
		if goodrun(total_reward):
			streak = streak + 1
			print("good run! streak is %d" % streak)
		else:
			streak = 0
			print("broke streak! :(")

	return Q, trial 	# done when 5 successful trials in a row

# slim version of best_seq that just finds the total reward of the max path through Q. If matrix guides to invalid moves, return a low reward (-1)
def check_seq(Q, T, R):
	current_state = STARTSTATE
	total_reward = 0
	while(current_state != ENDSTATE):
		action = rand_idx_max(Q[current_state]) # index of max value in Q[current_state]
		total_reward = total_reward + R[current_state][action]	# add latest reward to total reward
		current_state = T[current_state][action]
		if current_state == -1:
			return -1
	return total_reward

# Q-learning process to iteratively create Q matrix
# Stops when Q is fully converged: when every cell in Q matrix is within 1E-5 of that cell [blocksize] trials ago		
def fullconvergeQ(R, T):
	print("Starting learning iteration...")
	Q = [[-1 for j in range(NACTIONS)] for i in range(NSTATES)] # Initialize matrix Q to 0s

	# Loop in blocks to efficiently check for convergence every so often
	oldQ = None
	blk = 0
	trial = 0
	while(not converged(oldQ, Q)):
		print("Starting block %d (trial %d). Current Q:" % (blk, trial))
		# printmat(compmat(oldQ, Q))
		# printmat(Q)
		oldQ = copy.deepcopy(Q)
		# Main loop for each episode
		for it in range(BLOCKSIZE):						# Trials are batched for speed
			eps = max(0.5, 1/(trial+1))
			print(eps)
			print(trial)
			# current_state = random.randrange(NSTATES)	# Each episode starts from a random state
			current_state = STARTSTATE					# Each episode starts from starting state
			# action_seq = []							# Sequence of actions the robot takes in this trial
			# total_reward = 0						
			while (current_state != ENDSTATE):	# The robot keeps acting until it reaches end state. ! Maybe we don't need predefined end states
				action = exploitexplore(Q[current_state], eps)	# biased choice favoring good options with some exploration
				# action = random.randrange(NACTIONS)		# chooses next action randomly
				# action = rand_idx_max(Q[current_state])	# chooses best next action (better than random). But doesn't explore, so might miss better possibilities
				next_state = T[current_state][action]

				# Q formula
				Q[current_state][action] = R[current_state][action] + GAMMA * (max(Q[next_state]) if next_state != -1 else -1)

				if next_state != -1:
					current_state = next_state # if robot tries an action that doesn't lead anywhere, state does not change (self loop)

				# action_seq.append(action)
				# total_reward = total_reward + R[current_state][action]	# add latest reward to total reward

			# end of trial
			trial = trial + 1
			# sys.stdout.write("Trial %d. Path: " % (it + BLOCKSIZE*blk))
			# print(state_seq)
			# sys.stdout.write(" Total Reward: %d\n" % total_reward)

		# end of block
		print("end of block %d" % blk)
		blk = blk + 1
		
	return Q, trial

# returns a list of the best path through Q (ordered list of states) (not including start state, since there is no action to get there). Could alternatively return an ordered series of actions.
def best_seq(Q, T, A, R):
	current_state = STARTSTATE
	actions = []
	total_reward = 0
	path = [current_state]
	while(current_state != ENDSTATE):
		action = rand_idx_max(Q[current_state]) # index of max value in Q[current_state]
		actions.append(A[action])
		total_reward = total_reward + R[current_state][action]	# add latest reward to total reward
		current_state = T[current_state][action]
		if current_state == -1:
			print("Warning: Q matrix picked invalid action!")
		path.append(current_state)
	return actions, path, total_reward


# Ordered list of actions
A = ["grab dowel",
	 "grab front bracket",
	 "grab back bracket",
	 "grab seat",
	 "grab top bracket",
	 "grab long dowel",
	 "grab back",
	 "hold"]

# Reward Matrix: (s,a) --> r
R = [[10, 10, 10, -1, 10, -1, -1, -1],
	 [-1, 10, 10, -1, 10, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, 10, 10, -1, -1, -1, -1, -1],
	 [10, 10, 10, -1, -1, -1, -1, -1],
	 [-1, 10, 10, -1, -1, -1, -1, -1],
	 [-1, 10, 10, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, 10, -1, -1, -1, -1, -1],
	 [10, 10, -1, -1, -1, -1, -1, -1],
	 [10, 10, 10, -1, -1, -1, -1, -1],
	 [-1, -1, 10, -1, -1, -1, -1, -1],
	 [-1, 10, -1, -1, -1, -1, -1, -1],
	 [-1, 10, 10, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, 10, -1, -1, -1, -1, -1],
	 [10, 10, -1, -1, -1, -1, -1, -1],
	 [-1, -1, 10, -1, -1, -1, -1, -1],
	 [-1, 10, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, 10, -1, -1, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 10],
	 [10, -1, -1, -1, 10, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, 10, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, -1, 10, -1, -1, -1],
	 [-1, -1, -1, -1, -1, 10, 10, -1],
	 [-1, -1, -1, -1, -1, -1, 10, -1],
	 [-1, -1, -1, -1, -1, 10, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 10],
	 [10, 10, 10, -1, -1, -1, -1, -1],
	 [-1, 10, 10, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, 10, 10, -1, -1, -1, -1, -1],
	 [10, 10, 10, -1, -1, -1, -1, -1],
	 [-1, 10, 10, -1, -1, -1, -1, -1],
	 [-1, 10, 10, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, 10, 10, -1, -1, -1, -1, -1],
	 [10, 10, 10, -1, -1, -1, -1, -1],
	 [10, 10, 10, -1, -1, -1, -1, -1],
	 [-1, -1, 10, -1, -1, -1, -1, -1],
	 [-1, 10, -1, -1, -1, -1, -1, -1],
	 [-1, 10, 10, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, 10, -1, -1, -1, -1, -1],
	 [10, 10, -1, -1, -1, -1, -1, -1],
	 [-1, -1, 10, -1, -1, -1, -1, -1],
	 [-1, 10, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, 10, -1, -1, -1, -1],
	 [-1, -1, -1, -1, -1, -1, -1, 10],
	 [-1, -1, -1, -1, 10, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [10, -1, -1, -1, 10, -1, -1, -1],
	 [10, -1, -1, -1, -1, -1, -1, -1],
	 [-1, -1, -1, -1, 10, -1, -1, -1],
	 [-1, -1, -1, -1, -1, 10, 10, -1],
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

# Q, ntrials = fullconvergeQ(R, T)
Q, ntrials = goodenoughQ(R, T)
best_actions, best_path, total_reward = best_seq(Q, T, A, R)
print("Done learning. Q:")
printmat(Q)
print("N trials: %d" % ntrials)
print("Total reward: %d" % total_reward)
sys.stdout.write("Optimal actions: ")
print(best_actions)
sys.stdout.write("State sequence of those: ")
print(best_path)


# Two major conceptual problems:
# 1. What to do after bad action. It doesn't advance trials. Options:
#		Use preferences to allow trials to iterate -- I think this is best
#		Keep trying until robot gets it, adding rewards in each time. Takes forever and does all learning in one trial
#		Have human manually advance state by taking object themself. Additionally could send in reward for desired action. But then human is just telling robot what to do next time.
					
