import tensorflow as tf
import numpy as np
char_rdic = [x for x in "abcdefghijklmnopqrstuvwxyz"] # id -> char
char_dic = {w: i for i, w in enumerate(char_rdic)} # char -> id

def char_to_one_hot_v(c):
    alpha_logits_base = [0]*26
    alpha_logits_base[char_dic[c]]=1
    return alpha_logits_base

x_data = np.array([ char_to_one_hot_v('h'),
                    char_to_one_hot_v('e'), # e
                    char_to_one_hot_v('l'), # l
                    char_to_one_hot_v('l')], # l
                    dtype='f')

sample = [char_dic[c] for c in "hello"] # to index
# Configuration
char_vocab_size = len(char_dic)
rnn_size = char_vocab_size # 1 hot coding (one of 4)
time_step_size = 4 # 'hell' -> predict 'ello'
batch_size = 1 # one sample
# RNN model
rnn_cells = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cells.state_size])
X_split = tf.split(0, time_step_size, x_data)
outputs, state = tf.nn.rnn(rnn_cells, X_split, state)
# logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols].
# targets: list of 1D batch-sized int32 Tensors of the same length as logits.
# weights: list of 1D batch-sized float-Tensors of the same length as logits.
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * batch_size])
loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    for i in range(500):
        sess.run(train_op)
        print(sess.run(logits))
        result = sess.run(tf.arg_max(logits, 1))
        print (result, [char_rdic[t] for t in result]) 