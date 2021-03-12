from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

def prepare_model_settings(stock_number,data_out_number,proc_days,\
                           hidden1,hidden2):
  data_input_number = proc_days * stock_number
  return {
    'data_in_number': data_input_number,
    'data_out_number': data_out_number,
    'hidden1_units': hidden1,
    'hidden2_units': hidden2,
    }

def create_model(data_inputs, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  if model_architecture == 'line':
    return create_stock_line_model(data_inputs, model_settings,
                                  is_training)
  elif model_architecture == 'relu':
    return create_stock_relu_model(data_inputs, model_settings, is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "line", "relu"')
    
def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)

def create_stock_line_model(data_inputs, model_setting, is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  data_in_number = model_setting['data_in_number']
  data_out_number = model_setting['data_out_number']
  weights = tf.Variable(
      tf.truncated_normal([data_in_number, data_out_number], stddev=0.001))
  bias = tf.Variable(tf.zeros([data_out_number]))
  result = tf.matmul(data_inputs, weights) + bias
  if is_training:
    return result, dropout_prob
  else:
    return result

def create_stock_relu_model(data_inputs, model_setting, is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  
  data_in_number = model_setting['data_in_number']
  data_out_number = model_setting['data_out_number']
  
  hidden1_units = model_setting['hidden1_units']
  hidden2_units = model_setting['hidden2_units']
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([data_in_number, hidden1_units],
                            stddev=1.0 / math.sqrt(float(data_in_number))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(data_inputs, weights) + biases)
  if is_training:
    first_dropout = tf.nn.dropout(hidden1, dropout_prob)
  else:
    first_dropout = hidden1
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(first_dropout, weights) + biases)
    
  if is_training:
    second_dropout = tf.nn.dropout(hidden2, dropout_prob)
  else:
    second_dropout = hidden2
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, data_out_number],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([data_out_number]),
                         name='biases')
    result = tf.matmul(second_dropout, weights) + biases
    
  if is_training:
    return result, dropout_prob
  else:
    return result
