#!/usr/bin/env python2

import os
import argparse
import sys
import pickle
import distutils.dir_util
from threading import Lock

import numpy as np
import random
from sklearn.externals import joblib
import rospy
from rospy import init_node, is_shutdown
from std_msgs.msg import String
from std_srvs.srv import Empty
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

from human_robot_collaboration.controller import BaseController
from svox_tts.srv import Speech, SpeechRequest
# from hrc_pred_supp_bhv.service_request import ServiceRequest, finished_request
from hrc_pred_supp_bhv.task_def import *
from hrc_pred_supp_bhv.srv import *
from hrc_pred_supp_bhv.bern_hmm.bernoulli_hmm import *


NSTATES = 71    # number of states
NACTIONS = 8    # number of actions
STARTSTATE = 0  # starting state, if we even care
ENDSTATE = 70   # end state, assuming are using a predefined end
GAMMA = .3      # Q learning discount factor for future rewards


parser = argparse.ArgumentParser("Run the autonomous controller for bern_hmm_preds")
parser.add_argument('-path', '--path',
                    help='path to the model files',
                    default=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models')))
parser.add_argument('-ns', '--n_states',
                    help='number of HMM hidden states',
                    default=128)
parser.add_argument('-d', '--delay',
                    help='time delay over which to make predictions',
                    default=1)
parser.add_argument('-uidx', '--user_idx',
                    help='index of user preferences as defined in dict',
                    default=0)
parser.add_argument('-tfl', '--transfer_learning',
                    help='whether to test transfer learning task, '
                         'and how difficult the tf task should be: '
                         '0 --> no tf, 1 --> easy tf, 2 --> hard tf',
                    choices=['0', '1', '2'],
                    default=0)
parser.add_argument('-p', '--participant', help='id of participant', default='test')


class QController(BaseController):

    OBJECT_IDX = {              # parameters to pass to _____ in BaseController to pick up each object
        "seat_1":                [(BaseController.LEFT, 198),0],
        "back_1":                [(BaseController.LEFT, 201),0],
        "dowel_1":               [(BaseController.LEFT, 150),0],
        "dowel_2":               [(BaseController.LEFT, 151),0],
        "dowel_3":               [(BaseController.LEFT, 152),0],
        "dowel_4":               [(BaseController.LEFT, 153),0],
        "dowel_5":               [(BaseController.LEFT, 154),0],
        "dowel_6":               [(BaseController.LEFT, 155),0],
        "long_dowel_1":          [(BaseController.LEFT, 156),0],
        "front_bracket_1":       [(BaseController.RIGHT, 14),0],
        "front_bracket_2":       [(BaseController.RIGHT, 15),0],
        "back_bracket_1":        [(BaseController.RIGHT, 18),0],
        "back_bracket_2":        [(BaseController.RIGHT, 19),0],
        "top_bracket_1":         [(BaseController.RIGHT, 16),0],
        "top_bracket_2":         [(BaseController.RIGHT, 17),0],
        "screwdriver_1":         [(BaseController.RIGHT, 20),0],
    }

    BRING = 'get_pass'
    CLEAR = 'cleanup'
    HOLD_LEG = 'hold_leg'
    HOLD_TOP = 'hold_top'
    HOLD = 'hold_leg'
    WEB_INTERFACE = '/hrc_learning/web_interface/pressed'
    OBS_TALKER = '/hmm_bern_preds/obs'
    TTS_DISPLAY = '/svox_tts/speech_output'
    TTS_SERVICE = '/svox_tts/speech'

    Inventory = {
        "seat":             1,
        "back":             1,
        "dowel":            6,
        "long_dowel":       1,
        "front_bracket":    2,
        "back_bracket":     2,
        "top_bracket":      2,
    }

    # Transition Matrix: (s,a) --> s'
    # THIS IS NOT DOMAIN KNOWLEDGE. It's a shorthand for keeping track of the state of the world.
    # Every row is a state (vector of objects on the table) and each column is an action that changes that state.
    # The -1 entries are actions that would lead to out-of-bound states (for sanity, I only mapped the states
    # reachable in the task model and just ensured that the human player follow the rules of the task)
    # This is easily expandable to a system where the robot keeps track of the objects on the table and knows how each action changes them (since almost all actions simply add an object)
    # Such a system allows the human player to set their own rules and teach the robot to assist in any task.
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

    def __init__(self, path, n_states, delay, user_idx, transfer_learning, timer_path=None, **kwargs):
        super(QController, self).__init__(
            left=True,      # initialize fields in BaseController
            right=True,
            speech=True,
            listen=True,
            recovery=True,
            timer_path=os.path.join(path, timer_path),
            **kwargs)
        self.path = path
        self.n_states = int(n_states)
        self.delay = int(delay)
        self.user_idx = int(user_idx)
        self.transfer_learning = int(transfer_learning)
        self.time_step = 0
        self.traj_len = None
        self.model = None
        self.task_def = None
        self.obj_state = None
        self.last_brought = None
        self.current_ep_bin_feats = None #[[0]*len(self.OBJ_PRESENCE)]
        self.human_input = None
        rospy.loginfo('Waiting for speech service...')
        rospy.wait_for_service(self.TTS_SERVICE)
        self.tts_service = rospy.ServiceProxy(self.TTS_SERVICE, Speech)
        self.tts_display = rospy.Publisher(self.TTS_DISPLAY, String, queue_size=20)

    def web_interface_callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s ", data.data)
        self.human_input = data.data

    def reset_inv(self):
        self.Inventory["seat"] = 1
        self.Inventory["back"] = 1
        self.Inventory["dowel"] = 6
        self.Inventory["long_dowel"] = 1
        self.Inventory["front_bracket"] = 2
        self.Inventory["back_bracket"] = 2
        self.Inventory["top_bracket"] = 2

    def check_inv(self, action):
        if action == "br_seat" and self.Inventory["seat"] > 0:
            return 1
        if action == "br_back" and self.Inventory["back"] > 0:
            return 1
        if action == "br_dowel" and self.Inventory["dowel"] > 0:
            return 1
        if action == "br_long_dowel" and self.Inventory["long_dowel"] > 0:
            return 1
        if action == "br_front_bracket" and self.Inventory["front_bracket"] > 0:
            return 1
        if action == "br_top_bracket" and self.Inventory["top_bracket"] > 0:
            return 1
        if action == "br_back_bracket" and self.Inventory["back_bracket"] > 0:
            return 1
        if action == "hold":
            return 1
        return -1

    def update_inv(self, action):
        if action == "br_seat" and self.Inventory["seat"] > 0:
            self.Inventory["seat"] = self.Inventory["seat"] - 1
            return 1
        if action == "br_back" and self.Inventory["back"] > 0:
            self.Inventory["back"] = self.Inventory["back"] - 1
            return 1
        if action == "br_dowel" and self.Inventory["dowel"] > 0:
            self.Inventory["dowel"] = self.Inventory["dowel"] - 1
            return 1
        if action == "br_long_dowel" and self.Inventory["long_dowel"] > 0:
            self.Inventory["long_dowel"] = self.Inventory["long_dowel"] - 1
            return 1
        if action == "br_front_bracket" and self.Inventory["front_bracket"] > 0:
            self.Inventory["front_bracket"] = self.Inventory["front_bracket"] - 1
            return 1
        if action == "br_top_bracket" and self.Inventory["top_bracket"] > 0:
            self.Inventory["top_bracket"] = self.Inventory["top_bracket"] - 1
            return 1
        if action == "br_back_bracket" and self.Inventory["back_bracket"] > 0:
            self.Inventory["back_bracket"] = self.Inventory["back_bracket"] - 1
            return 1
        if action == "hold":
            return 1
        return -1

    # returns the index of one of the maxes in a list, chosen randomly
    def rand_idx_max(self, lst):
        m = max(lst)
        l = [i for i, v in enumerate(lst) if v == m]
        return random.choice(l)

    # returns the index of one of the positive values in a list, chosen randomly. If no positive values, choose randomly.
    def rand_idx_noneg(self, lst):
        l = [i for i, v in enumerate(lst) if v >= 0]
        if l == []:
            print("Warning: rand_idx_noneg picking from list with no non-negative values")
            return randrange(len(lst))
        return random.choice(l)

    def epsilondecreasing(self, options, trial, var):
        e = max(.1, (10-trial)/10.0) # 1, .9, .8, .7, .6, .5, .4, .3, .2, .1, .1, .1, .1, ...
        r = random.random()
        if r > e:                        # Exploit: pick (one of) the best
            print("EXPLOITING")
            return self.rand_idx_max(options)
        else:                            # Explore
            print("EXPLORING")
            unexplored = [i for i, v in enumerate(options) if v == 0]
            nonneg     = [i for i, v in enumerate(options) if v >= 0]
            if var == 'A':
                return random.choice(nonneg)    # A: pick randomly
            if unexplored:
                return random.choice(unexplored) # BC: try to pick from unexplored
            if var == 'B':
                return random.choice(nonneg) # B (pick randomly)
            return self.rand_idx_max(options)     # C (exploit after all)

    # Full program
    def _run(self):

        # self.rosbag_start()
        print("Please click a button on the web interface to confirm connection.")
        self.human_feedback()

        # speech test
        self._robot_speech(sentence='Ready to go!')

        # try to load a Q matrix
        print("checking for existing Q matrix file \'qload.pickle\'...")
        try:
            with open('qload.pickle', 'rb') as handle:
                Q = pickle.load(handle)
            print("Successfully loaded Q matrix:")
            print(Q)
        except:
            Q = [[0 for j in range(NACTIONS)] for i in range(NSTATES)] # Initialize matrix Q to 0s
            print("No existing Q matrix found; starting from scratch.")

        trial = 1

        print("Starting learning iterations...")

        while (True):    # keep running trials until user says to stop
            self._robot_speech(sentence='Are you ready?')
            greenlight = raw_input("Ready to start new trial. Continue? (enter 'y' to run a trial, anything else to quit)")
            if greenlight != 'y':
                break

            # begin trial
            self.reset_inv()
            eps = max(0.1, 1/(trial+1))
            current_state = STARTSTATE

            while (current_state != ENDSTATE):
                print("state: " + str(current_state))
                # begin one state-action step
                actionid = self.epsilondecreasing(Q[current_state], trial, 'A')  # 2. Use existing Q matrix to pick an action (exploit-explore)
                action = self.name_action(actionid)                             # string name of action

                if self.check_inv(action) < 0: # check that object is in inventory
                    print("considered" + action + "but that object is already used, applying negative feedback and trying something else")
                    Q[current_state][actionid] = -1
                    continue

                green, obj_taken_idx = self.take_action(action) # 3. Perform action up until human feedback

                if green:       # success
                    next_state = self.T[current_state][actionid]              # use T matrix to know next state. Equivalent of dead reckoning obj_state system. Better practice might be to put this T matrix in another file
#                    self.update_obj_state(green, action, object_action_idx) # assuming this updates mental map of what's on the table. I can maybe replace it with T matrix

                    if next_state == -1:
                        print("ERROR: Human accepted invalid action")
                        exit()
                        # !! in the future I could maybe eliminate T matrix and use my own obj state dict to keep track. But not central to project

                    self.update_inv(action)

                    # reward = 10
                    # self._robot_speech(sentence='How useful was that?')
                    reward = self.human_feedback()
                    if reward == 0:
                        print("Error: web feedback gave 0")
                        exit()

                    # Update Q matrix with positive reward
                    Q[current_state][actionid] = reward + GAMMA * max(Q[next_state])

                    current_state = next_state
                else:           # failure
                    print("Red button or grasper error")
                    # self._robot_speech(sentence='I will never do that again.')
                    # do not update state model or move through T matrix
                    Q[current_state][actionid] = -1 # no account for future because there is no future after an invalid move

                print("done with state-action step")
                print(Q)

            # save q matrix
            print("Storing Q matrix as \'q" + str(trial) + ".pickle\'")
            with open('q' + str(trial) + '.pickle', 'wb') as handle:
                pickle.dump(Q, handle, protocol=2)

            print("done with trial" + str(trial))
            trial = trial + 1
            self._robot_speech(sentence='Finished!')

        print("done with all trials (user decided to quit)")
        print("Rospy signal shutdown")
        self._robot_speech(sentence='Goodbye!')
        rospy.signal_shutdown("End of task.")


    # don't need to talk unless it's trivial
    def _robot_speech(self, sentence):
        self.tts_service(mode=5, string=sentence)
        self.tts_display.publish(sentence)
        rospy.sleep(0.2)

    # human to take any action online
    # use html page in folder to put my own buttons in
    # ! reponses published on a channel (continuously read)
    def human_feedback(self):
        print("Waiting for human feedback...")
        self.human_input = None
        rating = 0

        while self.human_input == None: # wait until meaningful input
            rospy.Subscriber(self.WEB_INTERFACE, String, self.web_interface_callback)
            rospy.rostime.wallsleep(0.5)
        if self.human_input != 'error':
            rating = self.human_input

        print("got human feedback: " + rating)
        if rating != '1' and rating != '2' and rating != '3' and rating != '4' and rating != '5' and rating != '6' and rating != '7' and rating != '8' and rating != '9' and rating != '10':
            print("ERROR: non-numeric rating received")
        rating = int(rating) # convert to int

        # if rating < 4:
        #     self._robot_speech(sentence='I\'ll try to find a better option next time.')
        # elif rating > 7:
        #     self._robot_speech(sentence='You\'re very welcome!')
        # else:
        #     self._robot_speech(sentence='I think I can do better.')

        return rating


    def name_action(self, actionid):
        if actionid == 0:
            return 'br_dowel'
        elif actionid == 1:
            return 'br_front_bracket'
        elif actionid == 2:
            return 'br_back_bracket'
        elif actionid == 3:
            return 'br_seat'
        elif actionid == 4:
            return 'br_top_bracket'
        elif actionid == 5:
            return 'br_long_dowel'
        elif actionid == 6:
            return 'br_back'
        elif actionid == 7:
            return 'hold'
        else:
            return 'invalid action id'

    # returns (1 if successful or 0 if failed (either with red button or 5 failed grasping attempts), obj_taken_idx)
    def take_action(self, action):
        #print rospy.get_name(), "I heard %s" % str(data.data)
        # I can actually leave this the way it is, because my "take action" is the same. The one thing that might vary is the end condition and feedback from human
        print("take_action got action ", action)
        obj_taken_idx = None

        for _ in range(5):  # Try taking action 4 times

            if action == 'cleanup':
                r = self._action(BaseController.RIGHT, (self.CLEAR, []), {'wait': True})
            elif action == 'br_screwdriver':
                obj_taken_idx = 1
                obj = action[3:]+'_'+str(obj_taken_idx)
                [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
                # if exists == 0:
                #     r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
                # else:
                #     return 2, obj_idx
            elif action == 'br_long_dowel':
                obj_taken_idx = 1
                obj = action[3:]+'_'+str(obj_taken_idx)
                [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
                # if exists == 0:
                #     r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
                # else:
                #     return 2, obj_idx
            elif action == 'br_front_bracket':
                obj_taken_idx = random.randint(1, 2)
                obj = action[3:]+'_'+str(obj_taken_idx)
                [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                total_objects = 2
                num_tries = 0
                while exists == 1 and num_tries < total_objects:
                    obj_taken_idx = random.randint(1, 2)
                    obj = action[3:]+'_'+str(obj_taken_idx)
                    [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                    num_tries += 1
                r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
            elif action == 'br_back_bracket':
                obj_taken_idx = random.randint(1, 2)
                obj = action[3:]+'_'+str(obj_taken_idx)
                [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                total_objects = 2
                num_tries = 0
                while exists == 1 and num_tries < total_objects:
                    obj_taken_idx = random.randint(1, 2)
                    obj = action[3:]+'_'+str(obj_taken_idx)
                    [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                    num_tries += 1
                r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
            elif action == 'br_top_bracket':
                obj_taken_idx = random.randint(1, 2)
                obj = action[3:]+'_'+str(obj_taken_idx)
                [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                total_objects = 2
                num_tries = 0
                while exists == 1 and num_tries < total_objects:
                    obj_taken_idx = random.randint(1, 2)
                    obj = action[3:]+'_'+str(obj_taken_idx)
                    [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                    num_tries += 1
                r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
            elif action == 'br_dowel':
                obj_taken_idx = random.randint(1, 6)
                obj = action[3:]+'_'+str(obj_taken_idx)
                [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                total_objects = 6
                num_tries = 0
                while exists == 1 and num_tries < total_objects:
                    obj_taken_idx = random.randint(1, 6)
                    obj = action[3:]+'_'+str(obj_taken_idx)
                    [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                    num_tries += 1
                r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
            elif action == 'hold':
                obj_taken_idx = random.randint(1, 6)
                obj = 'dowel_'+str(obj_taken_idx)
                [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAA ")
               # print(self.last_brought)
               # if self.last_brought['back'] == 1 or self.last_brought['long_dowel'] == 1:
               #     r = self._action(BaseController.RIGHT, (self.HOLD_TOP, [obj_idx]), {'wait': True})
               # else:
               #     r = self._action(BaseController.RIGHT, (self.HOLD_LEG, [obj_idx]), {'wait': True})
                r = self._action(BaseController.RIGHT, (self.HOLD_LEG, [obj_idx]), {'wait': True})    # simple hold
            elif action == 'br_seat':
                obj_taken_idx = 1
                obj = action[3:]+'_'+str(obj_taken_idx)
                [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
            elif action == 'br_back':
                obj_taken_idx = 1
                obj = action[3:]+'_'+str(obj_taken_idx)
                [(side, obj_idx), exists] = self.OBJECT_IDX[obj]
                r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
                # if exists == 0:
                #     r = self._action(side, (self.BRING, [obj_idx]), {'wait': True})
                # else:
                #     return 2, obj_idx
            else:
                raise ValueError('Unknown action: "{}".'.format(action))

            if r.success:           # ! Button responses # green button here
                self.OBJECT_IDX[obj][1] = 1
                return 1, obj_taken_idx
            # elif r.response == r.NO_OBJ:
            #     rospy.logerr(r.response)
            #     return 2, obj_taken_idx
            elif r.response == r.ACT_KILLED: # red button
                rospy.loginfo("Action failed: ACT_KILLED")
                #self.timer.log(message)
                return 0, obj_taken_idx
            elif r.response in (r.NO_IR_SENSOR, r.ACT_NOT_IMPL):
                rospy.logerr(r.response)
                self._stop()
            else:
                # Otherwise retry action
                rospy.logwarn('Retrying failed action {}. [{}]'.format(
                    action, r.response))
        return 0, obj_taken_idx

args = parser.parse_args()
controller = QController(
    path=args.path,
    n_states=args.n_states,
    delay=args.delay,
    user_idx=args.user_idx,
    transfer_learning=args.transfer_learning,
    timer_path='timer-{}.json'.format(args.participant)
)

if __name__ == '__main__':
    # rospy.init_node('HM`MBernPredsController')
    controller.time_step = 0
    controller.run()
