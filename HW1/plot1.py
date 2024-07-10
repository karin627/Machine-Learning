# plot fitting curve
import numpy as np
import matplotlib.pyplot as plt

"""
KEY_NAME : 
0 : song_duration
1 : acousticness
2 : danceability
3 : energy
4 : instrument
5 : key
6 : liveness
7 : loudness
8 : speechiness
9 : tempo
10 : audio_valence
"""

M =5
KEY = 2 # have to minus one
KEY_NAME = "danceability"
LAMBDA = '0'

# input training list
training_x = []
training_y = []
training_t = []

# for x in open('./results/training_x_'+str(KEY+1)+'.txt', 'r'):
for x in open('training_x_'+str(KEY+1)+'.txt', 'r'):
    training_x.append(float(x))
if LAMBDA=='0':
    # for y in open('./results/training_y_'+str(M)+'.txt', 'r'):
    for y in open('training_y_'+str(M)+'.txt', 'r'):
        training_y.append(float(y))
else:
    # for y in open('./results/training_y_'+str(M)+'_lambda'+LAMBDA+'.txt', 'r'):
    for y in open('training_y_'+str(M)+'_lambda'+LAMBDA+'.txt', 'r'):
        training_y.append(float(y))
# for t in open('./results/training_t.txt', 'r'):
for t in open('./results/training_t.txt', 'r'):
    training_t.append(float(t))

# sort training data
training_prediction_data = list(zip(training_x, training_y))
sorted_training_prediction_data = sorted(training_prediction_data, key=lambda pair: pair[0])
sorted_training_x = [pair[0] for pair in sorted_training_prediction_data]
sorted_training_y = [pair[1] for pair in sorted_training_prediction_data]


# plot training data
plt.figure(figsize=(6, 6))
plt.scatter(training_x, training_t, label='target', color='green', s=10)
plt.scatter(sorted_training_x, sorted_training_y, label='prediction', color='blue', s=7)
# fitting_curve = np.polyfit(sorted_training_x, sorted_training_y, 14)
# plt.plot(sorted_training_x, np.polyval(fitting_curve, sorted_training_x), label='fitting curve', color='red')
plt.xlabel('input')
plt.ylabel('output')
plt.legend()
plt.title('training(M = '+str(M)+'  KEY = '+str(KEY+1)+':'+KEY_NAME+'  λ = 0'+')')
plt.show()

# # input testing list
# testing_x = []
# testing_y = []
# testing_t = []
# # for x in open('./results/testing_x_'+str(KEY+1)+'.txt', 'r'):
# for x in open('testing_x_'+str(KEY+1)+'.txt', 'r'):
#     testing_x.append(float(x))
# if LAMBDA=='0':
#     # for y in open('./results/testing_y_'+str(M)+'.txt', 'r'):
#     for y in open('testing_y_'+str(M)+'.txt', 'r'):
#         testing_y.append(float(y))
# else:
#     # for y in open('./results/testing_y_'+str(M)+'_lambda'+LAMBDA+'.txt', 'r'):
#     for y in open('testing_y_'+str(M)+'_lambda'+LAMBDA+'.txt', 'r'):
#         testing_y.append(float(y))
# # for t in open('./results/testing_t.txt', 'r'):
# for t in open('testing_t.txt', 'r'):
#     testing_t.append(float(t))

# # sort testing data
# testing_prediction_data = list(zip(testing_x, testing_y))
# sorted_testing_prediction_data = sorted(testing_prediction_data, key=lambda pair: pair[0])
# sorted_testing_x = [pair[0] for pair in sorted_testing_prediction_data]
# sorted_testing_y = [pair[1] for pair in sorted_testing_prediction_data]

# # plot testing data
# plt.figure(figsize=(6, 6))
# plt.scatter(testing_x, testing_t, label='target', color='green', s=10)
# plt.scatter(sorted_testing_x, sorted_testing_y, label='prediction', color='blue', s=7)
# # fitting_curve = np.polyfit(sorted_testing_x, sorted_testing_y, 14)
# # plt.plot(sorted_testing_x, np.polyval(fitting_curve, sorted_testing_x), color='red', label='fitting curve')
# plt.xlabel('input')
# plt.ylabel('output')
# plt.legend()
# plt.title('testing(M = '+str(M)+'  KEY = '+str(KEY+1)+':'+KEY_NAME+'  λ = 0'+')')
# plt.show()