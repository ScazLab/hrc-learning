import pickle
import os
import sys


# Q matrix after first trial

#                                 dl  fb  bb  st  tb  ld  bk  hd
#                             ----------------------------------
# 0   nothing                 :    1,  0,  0,  0,  0,  0,  0,  0
# 1   1 dowel                 :   -1,  0,  5,  0,  0,  0, -1,  0  v----- seat in progress -----v
# 2   1 fb                    :    0,  0,  0,  0,  0,  0,  0,  0
# 3   1 bb                    :    0,  0,  0,  0,  0,  0,  0,  0
# 4   1 dl, 1 fb              :    0,  0,  0,  0,  0,  0,  0,  0
# 5   1 dl, 1 bb              :    0,  0, 10,  0,  0,  0,  0, -1
# 6   2 dl, 1 fb              :    0,  0,  0,  0,  0,  0,  0,  0
# 7   2 dl, 1 bb              :    0,  0,  0,  0,  0,  0,  0,  0
# 8   1 dl, 2 fb              :    0,  0,  0,  0,  0,  0,  0,  0
# 9   1 dl, 2 bb              :   10,  0,  0, -1,  0, -1,  0, -1
# 10  1 dl, 1 fb, 1 bb        :    0,  0,  0,  0,  0,  0,  0,  0
# 11  2 dl, 2 fb              :    0,  0,  0,  0,  0,  0,  0,  0
# 12  2 dl, 2 bb              :    3,  0,  0,  0,  0,  0,  0,  0
# 13  2 dl, 1 fb, 1 bb        :    0,  0,  0,  0,  0,  0,  0,  0
# 14  3 dl, 2 fb              :    0,  0,  0,  0,  0,  0,  0,  0
# 15  3 dl, 2 bb              :    0, 10,  0,  0,  0,  0,  0,  0
# 16  3 dl, 1 fb, 1 bb        :    0,  0,  0,  0,  0,  0,  0,  0
# 17  2 dl, 2 fb, 1 bb        :    0,  0,  0,  0,  0,  0,  0,  0
# 18  2 dl, 1 fb, 2 bb        :    0,  0,  0,  0,  0,  0,  0,  0
# 19  3 dl, 2 fb, 1 bb        :    0,  0,  0,  0,  0,  0,  0,  0
# 20  3 dl, 1 fb, 2 bb        :    0, 10,  0, -1, -1, -1, -1, -1
# 21  4 dl, 2 fb, 1 bb        :    0,  0,  0,  0,  0,  0,  0,  0
# 22  4 dl, 1 fb, 2 bb        :    0,  0,  0,  0,  0,  0,  0,  0
# 23  3 dl, 2 fb, 2 bb        :   10,  0,  0,  0, -1,  0,  0, -1
# 24  4 dl, 2 fb, 2 bb        :   -1,  0,  0, 10, -1, -1, -1, -1
# 25  4 dl, 2 fb, 2 bb, st    :   -1,  0,  0,  0,  0, -1, -1, 10
# 26  seat subassembly        :    5,  0,  0,  0,  0, -1, -1, -1  ^------------------------------^
# 27  1 tb                    :    0,  0,  0,  0,  0,  0,  0,  0  v------ back in progress ------v
# 28  1 dl, 1 tb              :    0,  0,  0,  0,  0,  0,  0,  0
# 29  1 dl, 2 tb              :    0,  0,  0,  0,  0,  0,  0,  0
# 30  2 dl, 1 tb              :    0,  0,  0,  0,  0,  0,  0,  0
# 31  2 dl, 2 tb              :    0,  0,  0,  0,  0,  0,  0,  0
# 32  2 dl, 2 tb, ld          :    0,  0,  0,  0,  0,  0,  0,  0
# 33  2 dl, 2 tb, bk          :    0,  0,  0,  0,  0,  0,  0,  0
# 34  2 dl, 2 tb, ld, bk      :    0,  0,  0,  0,  0,  0,  0, 10
# 35  back subassembly        :    0,  0,  0,  0,  0,  0,  0,  0  ^------------------------------^
# 36  bs, 1 dl                :    0,  0,  0,  0,  0,  0,  0,  0  v------ back subassembly + seat in progress ------v
# 37  bs, 1 fb                :    0,  0,  0,  0,  0,  0,  0,  0
# 38  bs, 1 bb                :    0,  0,  0,  0,  0,  0,  0,  0
# 39  bs, 1 dl, 1 fb          :    0,  0,  0,  0,  0,  0,  0,  0
# 40  bs, 1 dl, 1 bb          :    0,  0,  0,  0,  0,  0,  0,  0  ###
# 41  bs, 2 dl, 1 fb          :    0,  0,  0,  0,  0,  0,  0,  0
# 42  bs, 2 dl, 1 bb          :    0,  0,  0,  0,  0,  0,  0,  0
# 43  bs, 1 dl, 2 fb          :    0,  0,  0,  0,  0,  0,  0,  0
# 44  bs, 1 dl, 2 bb          :    0,  0,  0,  0,  0,  0,  0,  0
# 45  bs, 1 dl, 1 fb, 1 bb    :    0,  0,  0,  0,  0,  0,  0,  0
# 46  bs, 2 dl, 2 fb          :    0,  0,  0,  0,  0,  0,  0,  0
# 47  bs, 2 dl, 2 bb          :    0,  0,  0,  0,  0,  0,  0,  0
# 48  bs, 2 dl, 1 fb, 1 bb    :    0,  0,  0,  0,  0,  0,  0,  0  ### somehow these two rows are generating high Q values in col 1 (fb). Some future reward is accumulating there.
# 49  bs, 3 dl, 2 fb          :    0,  0,  0,  0,  0,  0,  0,  0
# 50  bs, 3 dl, 2 bb          :    0,  0,  0,  0,  0,  0,  0,  0
# 51  bs, 3 dl, 1 fb, 1 bb    :    0,  0,  0,  0,  0,  0,  0,  0
# 52  bs, 2 dl, 2 fb, 1 bb    :    0,  0,  0,  0,  0,  0,  0,  0
# 53  bs, 2 dl, 1 fb, 2 bb    :    0,  0,  0,  0,  0,  0,  0,  0
# 54  bs, 3 dl, 2 fb, 1 bb    :    0,  0,  0,  0,  0,  0,  0,  0
# 55  bs, 3 dl, 1 fb, 2 bb    :    0,  0,  0,  0,  0,  0,  0,  0
# 56  bs, 4 dl, 2 fb, 1 bb    :    0,  0,  0,  0,  0,  0,  0,  0
# 57  bs, 4 dl, 1 fb, 2 bb    :    0,  0,  0,  0,  0,  0,  0,  0
# 58  bs, 3 dl, 2 fb, 2 bb    :    0,  0,  0,  0,  0,  0,  0,  0
# 59  bs, 4 dl, 2 fb, 2 bb    :    0,  0,  0,  0,  0,  0,  0,  0
# 60  bs, 4 dl, 2 fb, 2 bb, st:    0,  0,  0,  0,  0,  0,  0, 10  ^-------------------------------------------------^
# 61  ss, 1 dl                :    0,  0,  0,  0, 10, -1,  0,  0  v------ seat subassembly + back in progress ------v
# 62  ss, 1 tb                :    0,  0,  0,  0,  0,  0,  0,  0
# 63  ss, 1 dl, 1 tb          :    0,  0,  0,  0, 10,  0, -1, -1
# 64  ss, 1 dl, 2 tb          :   10,  0,  0,  0,  0,  0,  0,  0
# 65  ss, 2 dl, 1 tb          :    0,  0,  0,  0,  0,  0,  0,  0
# 66  ss, 2 dl, 2 tb          :    0,  0,  0,  0,  0,  0, 10, -1
# 67  ss, 2 dl, 2 tb, ld      :    0,  0,  0,  0,  0,  0,  0,  0
# 68  ss, 2 dl, 2 tb, bk      :    0,  0,  0,  0,  0, 10,  0,  0
# 69  ss, 2 dl, 2 tb, ld, bk  :    0,  0,  0,  0,  0,  0,  0, 10  ^-------------------------------------------------^
# 70  chair                   :    0,  0,  0,  0,  0,  0,  0,  0



Q =  [[  1,  0,  0,  0,  0,  0,  0,  0 ],
      [ -1,  0,  5,  0,  0,  0, -1,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0, 10,  0,  0,  0,  0, -1 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [ 10,  0,  0, -1,  0, -1,  0, -1 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  3,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0, 10,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0, 10,  0, -1, -1, -1, -1, -1 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [ 10,  0,  0,  0, -1,  0,  0, -1 ],
      [ -1,  0,  0, 10, -1, -1, -1, -1 ],
      [ -1,  0,  0,  0,  0, -1, -1, 10 ],
      [  5,  0,  0,  0,  0, -1, -1, -1 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0, 10 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0, 10 ],
      [  0,  0,  0,  0, 10, -1,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0, 10,  0, -1, -1 ],
      [ 10,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0, 10, -1 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0, 10,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0, 10 ],
      [  0,  0,  0,  0,  0,  0,  0,  0 ]]


# trial = 'load'

# print("Storing Q matrix as \'q" + str(trial) + ".pickle\'")
# with open('q' + str(trial) + '.pickle', 'wb') as handle:
#     pickle.dump(Q, handle, protocol=2)


with open('qload.pickle', 'rb') as handle:
    P = pickle.load(handle)
print("Successfully loaded Q matrix:")
print(P)