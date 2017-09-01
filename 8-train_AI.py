
# coding: utf-8

# In[1]:

############ To use tensorboard, include all the summary op code below, and type: 
# tensorboard --logdir=/tmp/logs_path
# to launch type http://0.0.0.0:6006/ in a search bar 
# to kill ps -e | grep tensorboard in terminal
# the first number is the PID, type kill PID

import tensorflow as tf
import numpy as np
import argparse
import time
import subprocess as sp
import os
import hickle as hkl
print tf.__version__

num_layers = 4
drop_out_prob = 0.8
batch_size = 30
epochs = 30
learning_rate = 0.00001
test_size = 800
validation_size = 600
# In[40]:

videos_loaded = 0
for i in os.listdir(os.getcwd()):
    if i.endswith(".hkl"):
        if 'features' in i:
            print i
            if videos_loaded == 0:
                loaded = hkl.load(i)
            else:
                loaded = np.concatenate((loaded,hkl.load(i)), axis = 0)
            videos_loaded = videos_loaded + 1
            print loaded.shape


# In[41]:

videos_loaded = 0
for i in os.listdir(os.getcwd()):
    if i.endswith(".hkl"):
        if "labels" in i:
            print i
            if videos_loaded == 0:
                labels = hkl.load(i)
            else:
                labels = np.concatenate((labels,hkl.load(i)), axis = 0)
            videos_loaded = videos_loaded + 1
            print labels.shape
## Orders match up! Thats good.



def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p,:,:], b[p,:]

#reorder identically.
loaded, labels = unison_shuffled_copies(loaded,labels)
print loaded.shape, labels.shape

## Take the first 'test_size' examples as a test set. 
test_set = loaded[:test_size]
test_labels = labels[:test_size]
## Take indices from end of test_set to end of validation set size as our validation set
validation_set = loaded[test_size:(validation_size+test_size)]
validation_labels = labels[test_size:(validation_size+test_size)]
test_set_size = len(test_set)
## Now cut off the first 'test_size' + valid set numbers examples, because those are the test and validation sets
loaded = loaded[(test_size+validation_size):]
labels = labels[(test_size+validation_size):]
print "Test Set Shape: ", test_set.shape
print "Validation Set Shape: ", validation_set.shape
print "Training Set Shape: ", loaded.shape

## Save our test to test when we load the model we've trained. 
hkl.dump(test_set, 'test_data.hkl', mode='w', compression='gzip', compression_opts=9)
hkl.dump(test_labels, 'test_lbls.hkl', mode='w', compression='gzip', compression_opts=9)

device_name = "/gpu:0"

with tf.device(device_name):

    tf.reset_default_graph()


    logs_path = '/tmp/4_d-0.8'
    
    ## How often to print results and save checkpoint.
    display_step = 40

    # Network Parameters
    n_input = 2048 # 2048 long covolutional feature vector
    n_hidden = 1024 # hidden layer numfeatures
    n_classes = 3 # There are 3 possible outcomes, left, together, right
    

    # tf Graph input
    x = tf.placeholder("float32", [None, None, n_input]) ## this is [batch_size, n_steps, len_input_feature_vector]
    y = tf.placeholder("float32", [None, n_classes])
    input_batch_size = tf.placeholder("int32", None)

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }


    def RNN(x, weights, biases):

        
        cell = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)
        ## Variational recurrent = True means the same dropout mask is applied at each step. 
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=drop_out_prob)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        init_state = cell.zero_state(input_batch_size, tf.float32)
        outputs, states = tf.nn.dynamic_rnn(cell,x, initial_state = init_state, swap_memory = True) # swap_memory = True is incredibly important, otherwise gpu will probably run out of RAM
        print states
        # Linear activation, outputs -1 is the last 'frame'.
        return tf.matmul(outputs[:,-1,:], weights['out']) + biases['out']







    with tf.name_scope('Model'):
        pred = RNN(x, weights, biases)
    print "prediction", pred
    # Define loss and optimizer
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    with tf.name_scope('SGD'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    with tf.name_scope('Accuracy'):
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("training_accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()


# In[ ]:

current_epochs = 0
# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    saver = tf.train.Saver()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    step = 0
    
    
        
    # Keep training until we don't have any more data
    while step < (len(labels)/batch_size):
        batch_x = loaded[step*batch_size:(step+1)*batch_size]
        batch_y = labels[step*batch_size:(step+1)*batch_size]
       
        # x is in form batch * frames * conv_feature_size
        # y is in form batch * labels
        # Run optimization op (backprop)
        _,acc,loss,summary = sess.run([optimizer,accuracy,cost,merged_summary_op], feed_dict={x: batch_x, y: batch_y, input_batch_size:batch_size})
        print 'ran'
        ## The divide/multiply makes sense because it is reducing it to an int, i.e getting the whole number, could also cast as int, but this shows intent better.
        summary_writer.add_summary(summary, step*batch_size+current_epochs * (len(labels)/batch_size)*batch_size)

        if step % display_step == 0:
            # Calculate batch accuracy

            # calculate validation-set accuracy, because the GPU RAM is too small for such a large batch (the entire validation set, feed it through bit by bit)
            accuracies = []
            train_drop_out_prob = drop_out_prob
            drop_out_prob = 1.0
            for i in range(0,validation_size/batch_size):
                validation_batch_data = validation_set[i*batch_size:(i+1)*batch_size]
                validation_batch_labels = validation_labels[i*batch_size:(i+1)*batch_size]
                validation_batch_acc,_ = sess.run([accuracy,cost], feed_dict={x: validation_batch_data, y: validation_batch_labels, input_batch_size: batch_size})    
                accuracies.append(validation_batch_acc)
                

            # Create a new Summary object with your measure and add the summary
            summary = tf.Summary()
            summary.value.add(tag="Validation_Accuracy", simple_value=sum(accuracies)/len(accuracies))
            summary_writer.add_summary(summary, step*batch_size +current_epochs * (len(labels)/batch_size)*batch_size)
            # Save a checkpoint
            saver.save(sess, 'fencing_AI_checkpoint')
            
            print "Validation Accuracy - All Batches:", sum(accuracies)/len(accuracies)
            
            ## Set dropout back to regularizaing
            drop_out_prob = train_drop_out_prob
                
            print "Iter " + str(step*batch_size+current_epochs * (len(labels)/batch_size)*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Train Accuracy= " + \
                  "{:.5f}".format(acc)
                    
            


        step += 1
        if current_epochs < epochs:
            if step >= (len(labels)/batch_size):
                print "###################### New epoch ##########"
                current_epochs = current_epochs + 1
                learning_rate = learning_rate- (learning_rate*0.15)
                step = 0

                # Randomly reshuffle every batch.
                loaded, labels = unison_shuffled_copies(loaded,labels)

        
    print "Learning finished!"

  





