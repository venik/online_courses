import os

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate = 0.1
epochs = 200

base_path = '../Ex_Files_TensorFlow/Exercise Files/03'
test_file = base_path + '/sales_data_test.csv'
training_file = base_path + '/sales_data_training.csv'
inference_column = 'total_earnings'

input_size = 9
output_size = 1
layer1_size = 200

Input_scaler = MinMaxScaler()
Output_scaler = MinMaxScaler()

# prepare training data
training_df = pd.read_csv(training_file, dtype=float)
training_data = training_df.drop(inference_column, axis=1).values
training_output = training_df[[inference_column]].values

# prepare test data
test_df = pd.read_csv(test_file, dtype=float)
test_data = test_df.drop(inference_column, axis=1).values
test_output = test_df[[inference_column]].values

if True:
    # scale
    training_data_scaled = Input_scaler.fit_transform(training_data)
    training_output_scaled = Output_scaler.fit_transform(training_output)
    test_data_scaled = Input_scaler.transform(test_data)
    test_output_scaled = Output_scaler.transform(test_output)
else:
    training_data_scaled = training_data
    training_output_scaled = training_output
    test_data_scaled = test_data
    test_output_scaled = test_output

# input layer
with tf.variable_scope('input'):
    X = tf.placeholder(dtype=tf.float32, shape=(None, input_size))

# hidden layer
with tf.variable_scope('layer1'):
    W1 = tf.get_variable('W1', shape=[input_size, layer1_size], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', shape=[layer1_size], initializer=tf.zeros_initializer())
    layer1_output = tf.nn.relu(tf.matmul(X, W1) + b1)

# output layer
with tf.variable_scope('output'):
    W_out = tf.get_variable('W_out', shape=[layer1_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
    b_out = tf.get_variable('b_out', shape=[output_size], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer1_output, W_out) + b_out

with tf.variable_scope('cost'):
    output = tf.placeholder(dtype=tf.float32, shape=(None, output_size))
    cost = tf.reduce_mean(tf.squared_difference(prediction, output))

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.variable_scope('status'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        session.run(optimizer, feed_dict={X: training_data_scaled, output: training_output_scaled})

        if epochs % 5 == 0:
            training_cost, training_summary = session.run(
                [cost, summary],
                feed_dict={X: training_data_scaled, output: training_output_scaled}
            )

            testing_cost, test_summary = session.run(
                [cost, summary],
                feed_dict={X: test_data_scaled, output: test_output_scaled}
            )

            print('Training pass: {} training cost: {} testing cost: {}'.format(epoch, training_cost, testing_cost))

    final_training_cost, final_trainig_summary = session.run(
        [cost, summary],
        feed_dict={X: training_data_scaled, output: training_output_scaled}
    )
    final_testing_cost, final_testing_summary = session.run(
        [cost, summary],
        feed_dict={X: test_data_scaled, output: test_output_scaled}
    )

    print('Complete')
    print('Final testing cost: {} training cost: {}'.format(final_training_cost, final_testing_cost))
