
import tensorflow as tf


buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
encoder_inputs, decoder_inputs, target_weights = [],[],[]
for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
    encoder_inputs.append(tf.placeholder(tf.float32, shape=[None,100],
                                              name="encoder{0}".format(i)))
for i in range(buckets[-1][1] + 1):
    decoder_inputs.append(tf.placeholder(tf.float32, shape=[None,100],
                                              name="decoder{0}".format(i)))
    target_weights.append(tf.placeholder(tf.float32, shape=[None,],
                                              name="weight{0}".format(i)))

targets = [decoder_inputs[i + 1]
               for i in range(len(decoder_inputs) - 1)]
size = 128
target_vocab_size = 10000
source_vocab_size = 8000
num_samples = 1000
w = tf.get_variable("proj_w", [size, target_vocab_size])
w_t = tf.transpose(w)
b = tf.get_variable("proj_b", [target_vocab_size])
output_projection = (w, b)

def sampled_loss(labels, inputs):  # bug fixed
    labels = tf.reshape(labels, [-1, 1])
    return tf.nn.sampled_softmax_loss(w_t, b, labels, inputs, num_samples,
                                      target_vocab_size)
#cell = tf.contrib.rnn.GRUCell(size)
# def seq2seq_f(encoder_inputs=None, decoder_inputs=None, do_decode=False):
#   return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
#       encoder_inputs, decoder_inputs, cell,
#       num_encoder_symbols=source_vocab_size,
#       num_decoder_symbols=target_vocab_size,
#       embedding_size=size,
#       output_projection=output_projection,
#       feed_previous=do_decode)
outputs, losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          encoder_inputs, decoder_inputs, targets,
          target_weights, buckets, lambda x, y: tf.contrib.legacy_seq2seq.basic_rnn_seq2seq( x, y, tf.contrib.rnn.GRUCell(size)),
          softmax_loss_function=sampled_loss)