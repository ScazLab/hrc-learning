#!/usr/bin/env python

import os
import argparse
import sys
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
#from hrc_pred_supp_bhv.service_request import ServiceRequest, finished_request
from hrc_pred_supp_bhv.task_def import *
from hrc_pred_supp_bhv.srv import *
from hrc_pred_supp_bhv.bern_hmm.bernoulli_hmm import *

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

class DummyPredictor(object):

    def __init__(self, object_list):
        self.obj = object_list
        self.words = [o.split('_')[0] for o in self.obj]

    @property
    def n_obj(self):
        return len(self.obj)

    def transform(self, utterances):
        return np.array([[w in u.lower() for w in self.words]
                         for u in utterances])

    def predict(self, Xc, Xs, exclude=[]):
        # return an object that is in context and which name is in utterance
        intersection = Xs * Xc
        intersection[:, [self.obj.index(a) for a in exclude]] = 0
        chosen = -np.ones((Xc.shape[0]), dtype='int8')
        ii, jj = intersection.nonzero()
        chosen[ii] = jj
        scr = self.obj.index('screwdriver_1')
        chosen[(chosen == -1).nonzero()[0]] = scr
        return [self.obj[c] for c in chosen]


class SuppBhvsHMMController(BaseController):

    OBJECT_IDX = {
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
    WEB_INTERFACE = '/hrc_pred_supp_bhv/web_interface/pressed'
    OBS_TALKER = '/hmm_bern_preds/obs'
    TTS_DISPLAY = '/svox_tts/speech_output'
    TTS_SERVICE = '/svox_tts/speech'
    ROSBAG_START = '/rosbag/start'
    ROSBAG_STOP = '/rosbag/stop'
    #PROVIDE_OBS_SRV_NAME = 'provide_obs'
    #PROVIDE_OBS_SRV_TYPE = ProvideObs
    #NUM_FEATS = len(OBJECT_DICT.keys())
    #NUM_FEATS = 6
    MAX_SUPP_BHVS = 1
    ACTION_PROB_THRESH = 0.5

    def __init__(self, path, n_states, delay, user_idx, transfer_learning, timer_path=None, **kwargs):
        super(SuppBhvsHMMController, self).__init__(
            left=True,
            right=True,
            speech=False,
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
        self.rosbag_start = rospy.ServiceProxy(self.ROSBAG_START, Empty)
        self.rosbag_stop = rospy.ServiceProxy(self.ROSBAG_STOP, Empty)

    def obs_from_dict(self, d):
        # return tuple([self.OBJECT_IDX[key][1]
        #     for key in sorted(self.OBJECT_IDX.keys())]
        print("MAKING OBS FROM DICT")
        obs = []
        for key in range(len(self.task_def.FEATURES)):
            obs.append(d[self.task_def.FEATURES[key]])
            print(obs[-1])
        return tuple(obs)
        # print("Keys ", sorted(d.keys()))
        # return tuple([d[key]
        #     for key in sorted(d.keys())])

    def reset_dict(self, d):
        for key in d.keys():
            d[key] = 0

    def prep_model(self):
        self.task_def = TaskDef(self.transfer_learning)
        self.traj_len = self.task_def.main_task.train_set_actions.shape[1]
        self.obj_state = {v: 0 for k, v in self.task_def.FEATURES.iteritems()}
        self.last_brought = {v: 0 for k, v in self.task_def.MAIN_OBJ.iteritems()}
        self.current_ep_bin_feats = []
        #self.current_ep_bin_feats = [[0]*len(self.obj_state)]
        # testing for user1
        train, train_lens = \
            self.task_def.prep_X(trajectories=
                                 self.task_def.main_task.train_set_actions)
        if self.transfer_learning == 0 or self.transfer_learning == 2:
            print("should be fine ",
                  self.task_def.main_task.train_set_sb_actions[self.user_idx].shape)
            train_sb, train_sb_lens = \
                self.task_def.prep_X(trajectories=
                                     self.task_def.main_task.train_set_sb_actions[self.user_idx])
            train_set_sb = self.task_def.main_task.train_set_sb
        else:
            print("should be fine ",
                  self.task_def.tf_task.train_set_sb_actions[self.user_idx].shape)
            train_sb, train_sb_lens = \
                self.task_def.prep_X(trajectories=
                                     self.task_def.tf_task.train_set_sb_actions[self.user_idx])
            train_set_sb = self.task_def.tf_task.train_set_sb
        print("should save here ", self.path)
        distutils.dir_util.mkpath(self.path)
        model_file = os.path.join(self.path, "BernoulliCondIndHMM.pkl")
        model = Model()
        if os.path.isfile(model_file) is False:
            model.train_hmm(self.path, train, train_lens, self.n_states)
            self.model = model.model
        else:
            self.model = joblib.load(model_file)

        return train_sb, train_sb_lens, train_set_sb

    def pred_supp_bhv(self, new_row_bin_feats, train_sb, train_sb_lens,
                      train_set_sb):
        self.current_ep_bin_feats.append(list(new_row_bin_feats))
        test_sb = np.array(self.current_ep_bin_feats)
        print("test_sb is ", test_sb)
        test_sb_lens = [test_sb.shape[0]]
        print("test_sb_lens is ", test_sb_lens)

        pred_supp_bhv = prob_supp_bhv_marg_hstate(model=self.model, delay=self.delay,
                                                  X_prepped_train=train_sb, X_prepped_test=test_sb,
                                                  supp_bhvs=train_set_sb[self.user_idx],
                                                  X_prepped_train_lengths=train_sb_lens,
                                                  X_prepped_test_lengths=test_sb_lens)
        print("pred_supp_bhv ", pred_supp_bhv)
        #max_prob_preds = pred_supp_bhv.max(axis=1)
        print("sort ", pred_supp_bhv.argsort(axis=1))
        argmax_prob_preds = pred_supp_bhv.argsort(axis=1)
        #argmax_prob_preds = pred_supp_bhv.argmax(axis=1)
        np.set_printoptions(precision=4, suppress=True)
        # print("train_set_sb[user_idx] \n", self.train_set_sb[self.user_idx])
        # print("predicting for test with gt \n", test_sb)
        # print("pred_supp_bhv \n", pred_supp_bhv)

        #print("max_prob_preds \n", max_prob_preds)
        print("argmax_prob_preds \n", argmax_prob_preds)
        print("len ", len(argmax_prob_preds[0]))
        print("Max item ", len(argmax_prob_preds[0])-1)

        robot_action_names = []
        for arg in reversed(argmax_prob_preds[-1]):
            if pred_supp_bhv[-1][arg] > self.ACTION_PROB_THRESH:
                robot_action_names.append(self.task_def.supp_bhvs_rev[arg])


        # # if 'br_dowel' is predicted, move it at the end of the array
        # if 'br_dowel' in robot_action_names:
        #     robot_action_names.remove('br_dowel')
        #     robot_action_names.append('br_dowel')


        # if 'hold' is predicted, move it at the end of the array
        if 'hold' in robot_action_names:
            robot_action_names.remove('hold')
            robot_action_names.append('hold')
        print("The predicted robot actions to be returned are ")
        print(robot_action_names)
        return robot_action_names

        # if test_sb[robot_action_idx] == 1:
        #     print("Correct supp bhv")
        # else:
        #     print("Incorrect supp bhv")

    # def run(self):
    #     rospy.Subscriber(self.OBS_TALKER, numpy_msg(Floats), self.take_action)
    #     rospy.spin()

    def web_interface_callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s ", data.data)
        self.human_input = data.data

    def _run(self):
        self.rosbag_start()
        train_sb, train_sb_lens, train_set_sb = self.prep_model()
        while not is_shutdown():
            print("Traj len ", self.traj_len)
            if self.time_step < self.traj_len-1:
                obs = self.obs_from_dict(self.obj_state)
                print("Obs from dict objects ")
                for key in range(19):
                    obj_name = self.task_def.FEATURES[key]
                    print("{} {}: {}".format(key, obj_name, self.obj_state[obj_name]))
                # for key in sorted(self.last_brought.keys()):
                #     print("{}: {}".format(key, self.last_brought[key]))
                print("!!!!!!!!!!!!!!!!!!!Obs from dict is ", obs)
                actions = self.pred_supp_bhv(obs, train_sb, train_sb_lens,
                                                train_set_sb)

                self.obj_state['hold_active'] = 0
                self.obj_state[self.task_def.FEATURES[self.task_def.OBJ_COUNT_IDX['long_dowel'][0]]] = 0
                self.obj_state[self.task_def.FEATURES[self.task_def.OBJ_COUNT_IDX['back'][0]]] = 0
                self.obj_state[self.task_def.FEATURES[self.task_def.OBJ_COUNT_IDX['seat'][0]]] = 0
                self.obj_state[self.task_def.FEATURES[self.task_def.OBJ_COUNT_IDX['type_back_bracket'][0]]] = 0
                self.obj_state[self.task_def.FEATURES[self.task_def.OBJ_COUNT_IDX['type_front_bracket'][0]]] = 0

                raw_input("Starting supp bhv series for time step {}".format(self.time_step))
                # self.pub.publish("saying stuffs")
                # rospy.sleep(2)
                #ServiceRequest(self.TTS_SERVICE, SpeechRequest.SAY, 'say stuffs', None)
                # self.tts_service(mode=5, string='We, robots, love you.')
                # rospy.sleep(2)

                # test = True
                # if test is True:
                #     actions = ['br_front_bracket', 'br_back_bracket']
                #     test = False

                for action in actions:
                    speak_action = action[3:]
                    if action == 'hold':
                        speak_action = action
                    print("speak_action ", speak_action)
                    speak_action_parts = speak_action.split('_')
                    if action != 'hold':
                        speak_action_parts.insert(0, 'bring')
                    speak_sentence = 'Performing robot predicted action'
                    for part in speak_action_parts:
                        speak_sentence = speak_sentence + ' ' + part
                    self._robot_speech(sentence=speak_sentence)
                    #self._robot_speech(sentence='Performing robot predicted action {}'.format(speak_action))
                    print("Now performing robot predicted action {} out of {}".format(action, actions))
                    #self.complete_robot_action('hold')
                    self.complete_robot_action(action)
                self._robot_speech(sentence='Please tell me what supportive behaviors I missed.')
                actions_requested = self.human_complete_action()
                print("Next phase performs any human requested actions ")
                for action in actions_requested:
                    print("Now performing human requested action {} out of {}".format(action, actions_requested))
                    self.complete_robot_action(action)
                self.time_step += 1
                raw_input("Move on to the next time step")
            else:
                self.rosbag_stop()
                rospy.signal_shutdown("End of task.")

    def _abort(self):
        self.rosbag_stop()

    def _robot_speech(self, sentence):
        self.tts_service(mode=5, string=sentence)
        self.tts_display.publish(sentence)
        rospy.sleep(1.5)

    def human_complete_action(self):
        raw_input("Take human action...")
        self.human_input = None
        actions_requested = []
        while self.human_input != 'next':
            rospy.Subscriber(self.WEB_INTERFACE, String, self.web_interface_callback)
            print("got human_input ", self.human_input)
            if self.human_input != None and 'br_'+self.human_input in self.task_def.SUPP_BHVS.keys() \
            and 'br_'+self.human_input not in actions_requested:
                actions_requested.append('br_'+self.human_input)
            if self.human_input == 'hold' and self.human_input not in actions_requested:
                actions_requested.append(self.human_input)
            rospy.rostime.wallsleep(0.5)
        print("got human_input next ")
        print("JUST FINISHED PREDICTION FOR TIME STEP ", self.time_step)
        return actions_requested

    def complete_robot_action(self, action):
        action_taken = 0

        print("STATE OF OBJECTS BEFORE PREDICTION ")
        for key in sorted(self.obj_state.keys()):
            print("{}: {}".format(key, self.obj_state[key]))

        # if action == 'br_scrdrv':
        #     action = 'br_screwdriver'
        # elif action == 'br_front_brackets':
        #     action = 'br_front_bracket'
        # elif action == 'br_back_brackets':
        #     action = 'br_back_bracket'

        #raw_input("Press Enter to continue... ")
        #action_taken = self.take_action(action)
        #self.reset_dict()
        #print("dict should be reset ", self.SUPP_BHVS)
        #print("TAKE ROBOT ACTION ")

        #action_taken = self.sim_take_action(action)
        action_taken, object_action_idx = self.take_action(action)
        print("action_taken ", action_taken)

        #self.reset_dict()
        print("action_taken ", action_taken)
        print("with br ", action)
        print("in dict ", action in self.obj_state)

        #self.sim_update_obj_state(action_taken, action)
        #print("Taken object_action_idx ", object_action_idx)
        if action_taken == 1:
            self.update_obj_state(action_taken, action, object_action_idx)
        else:
            print("This was an error and we keep track")

        print("dict should have something ")
        for key in sorted(self.obj_state.keys()):
            print("{}: {}".format(key, self.obj_state[key]))

        # print("last brought dict should have something ")
        # for key in sorted(self.last_brought.keys()):
        #     print("{}: {}".format(key, self.last_brought[key]))


        # raw_input("Take human action...")
        # self.human_input = None
        # while self.human_input != 'next':
        #     rospy.Subscriber(self.WEB_INTERFACE, String, self.web_interface_callback)
        #     print("got human_input ", self.human_input)
        #     if self.human_input != None and self.human_input != 'cleanup':
        #         if self.human_input in self.obj_state.keys():
        #             self.obj_state[self.human_input] = 1
        #     rospy.rostime.wallsleep(0.5)
        # print("got human_input next ")
        # print("JUST FINISHED PREDICTION FOR TIME STEP ", self.time_step)

        print("Move on to the next supp bhv for time step {}".format(self.time_step))


    def sim_update_obj_state(self, action):
        if action != 'cleanup' and action != 'hold' \
                and action in self.obj_state.keys():
            self.obj_state[action] = 1

    def update_obj_state(self, action_taken, action, object_action_idx):
        # check this
        # if self.last_brought['long_dowel'] == 1:
        #     self.obj_state[self.task_def.FEATURES[self.task_def.OBJ_COUNT_IDX['long_dowel'][0]]] = 0
        # if self.last_brought['back'] == 1:
        #     self.obj_state[self.task_def.FEATURES[self.task_def.OBJ_COUNT_IDX['back'][0]]] = 0
        # if self.last_brought['seat'] == 1:
        #     self.obj_state[self.task_def.FEATURES[self.task_def.OBJ_COUNT_IDX['seat'][0]]] = 0
        self.reset_dict(self.last_brought)
        print("action_taken ", action_taken)
        print("object_action_idx ", object_action_idx)
        # if object_action_idx != None:
        #     obj_in_dict = action[3:]+'_'+str(object_action_idx)
        # else:
        #     obj_in_dict = action[3:]
        obj_in_dict = action
        if object_action_idx != None and action != 'hold':
            obj_in_dict = action[3:]
        if action == 'hold':
            obj_in_dict = 'hold_active'
        print("dict obj ", obj_in_dict)
        if action_taken == 1 and action != 'cleanup' \
                and obj_in_dict in self.task_def.OBJ_COUNT_IDX.keys():
            # self.obj_state[obj_in_dict] = 1
            if obj_in_dict in self.last_brought.keys():
                self.last_brought[obj_in_dict] = 1
            feat_idx = self.task_def.OBJ_COUNT_IDX[obj_in_dict]
            print("*action ", action)
            print("*obj_in_dict ", obj_in_dict)
            if len(feat_idx) > 1:
                i = feat_idx[0]
                print("*i ", i)
                while(self.obj_state[self.task_def.FEATURES[i]]) == 1:
                    i += 1
                self.obj_state[self.task_def.FEATURES[i]] = 1
                print("*obj_state[i] ", self.obj_state[self.task_def.FEATURES[i]])
            else:
                self.obj_state[self.task_def.FEATURES[feat_idx[0]]] = 1
                print("*obj_state[feat_idx[0]] ", self.obj_state[self.task_def.FEATURES[feat_idx[0]]])
            if obj_in_dict == 'back_bracket':
                self.obj_state[self.task_def.FEATURES[self.task_def.OBJ_COUNT_IDX['type_back_bracket'][0]]] = 1
            if obj_in_dict == 'front_bracket':
                self.obj_state[self.task_def.FEATURES[self.task_def.OBJ_COUNT_IDX['type_front_bracket'][0]]] = 1


    def sim_take_action(self, action):
        print("Reached sim_take_action for action ", action)
        self.human_input = None
        while self.human_input != 'error' and self.human_input != 'home':
            rospy.Subscriber(self.WEB_INTERFACE, String, self.web_interface_callback)
            print("got human_input ", self.human_input)
            if self.human_input:
                self.sim_update_obj_state(self.human_input)
            rospy.rostime.wallsleep(0.5)
        if self.human_input == 'home':
            return 1
        if self.human_input == 'error':
            return 0

    def take_action(self, action):
        #print rospy.get_name(), "I heard %s" % str(data.data)
        print("take_action got action ", action)
        obj_taken_idx = None

        for _ in range(5):  # Try taking action 4 times
            # side, obj_1 = self.OBJECT_DICT['back_bracket_1']
            # _, obj_2 = self.OBJECT_DICT['back_bracket_2']
            # _, obj_3 = self.OBJECT_DICT['front_bracket_1']
            # _, obj_4 = self.OBJECT_DICT['front_bracket_2']
            #r = self._action(side, (self.BRING, [obj]), {'wait': True})

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
                print(self.last_brought)
                if self.last_brought['back'] == 1 or self.last_brought['long_dowel'] == 1:
                    r = self._action(BaseController.RIGHT, (self.HOLD_TOP, [obj_idx]), {'wait': True})
                else:
                    r = self._action(BaseController.RIGHT, (self.HOLD_LEG, [obj_idx]), {'wait': True})
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

            if r.success:
                self.OBJECT_IDX[obj][1] = 1
                return 1, obj_taken_idx
            # elif r.response == r.NO_OBJ:
            #     rospy.logerr(r.response)
            #     return 2, obj_taken_idx
            elif r.response == r.ACT_KILLED:
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
controller = SuppBhvsHMMController(
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