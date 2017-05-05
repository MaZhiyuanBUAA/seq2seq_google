#coding:utf-8
import os
import sys
import re

import tensorflow as tf

from config import FLAGS
import data_utils
from seq2seq_model_utils import create_model, get_predicted_sentence

def add_space(matched):
  intStr = matched.group('chinese')
  return intStr + ' '

def chat():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, forward_only=True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    sentence = re.sub(u'[^\u4e00-\u9fa5，。；：？！‘’“”、]','',sentence.decode('utf-8'))
    sentence = re.sub(u'(?P<chinese>[\u4e00-\u9fa5，。；：？！‘’“”、])',add_space,sentence)

    while sentence:
        predicted_sentence = get_predicted_sentence(sentence, vocab, rev_vocab, model, sess)
        print(predicted_sentence)
        print("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        sentence = re.sub(u'[^\u4e00-\u9fa5，。；：？！‘’“”、]','',sentence.decode('utf-8'))
        sentence = re.sub(u'(?P<chinese>[\u4e00-\u9fa5，。；：？！‘’“”、])',add_space,sentence)
