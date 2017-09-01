
# Run a pretrained_inception net over the clips to get a 1*2048 conv feature vector for each frame, save these. 
# thanks to https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/inception_cifar10/cifar-10_experiment.py
# for the example code which this is inspired by. 
import tensorflow as tf
import numpy as np
import os
from classify_image import *
import hickle as hkl
import time

FLAGS.model_dir = 'model/'
maybe_download_and_extract()
create_graph()


# Only way I could find of doing this in batches has large overheads, see https://github.com/tensorflow/tensorflow/issues/1021
## Therefore example by example
with tf.Session() as sess: 
    representation_tensor = sess.graph.get_tensor_by_name('pool_3:0')


    for i in os.listdir(os.getcwd() + "/preinception_data/"):
        if i.endswith(".hkl"):
            if "set" in i:
                number = i.split("-")[-1].replace(".hkl","")
                print number
                train_set = hkl.load(os.getcwd() + '/preinception_data/'+i)
                print "Training Data:", train_set.shape

                train_labels = hkl.load(os.getcwd() + '/final_training_data/'+"train_labels-" + str(number) + ".hkl")
                print "Train Labels:", train_labels.shape
                for example in range(len(train_set)):

                    
                    frame_representation = np.zeros((len(train_set[example]), 2048), dtype='float32')
                    start = time.time()
                    for frame in range(len(train_set[example])):
                        #start_frame = time.time()
                        # note that decode jpeg is a placeholder, it is where we put our input. 
                        rep = sess.run(representation_tensor, {'DecodeJpeg:0': train_set[example][frame]})
                        #print "Time for frame {} ", (time.time() - start_frame)
                        frame_representation[frame] = np.squeeze(rep)
                    
                    frame_representation = np.expand_dims(frame_representation,axis=0)    
                    print " ###########  Time for clip (21 forward passes) {} ", (time.time() - start)
                    
                    if example == 0:
                        data_set = frame_representation
                    else:
                        data_set = np.concatenate((data_set,frame_representation), axis = 0)
                   
                    
                    print data_set.shape
                hkl.dump(data_set, 'final_training_data/conv_features_train' + '-' + str(number) +'.hkl', mode='w', compression='gzip', compression_opts=9)
                
                print "Section Saved"

# In[3]:




# In[ ]:




# In[4]:

g = tf.get_default_graph()
names = [op.name for op in g.get_operations()]
print(names)


# In[18]:




# In[ ]:


        


# In[ ]:



