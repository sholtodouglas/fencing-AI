# Using Deep Learning to Referee Fencing

In fencing, when both competitors hit eachother within a few miliseconds, both sides of the scoreboard will light up. Then the referee has to make a decision on who's point it is. As a rough rule of thumb, it goes to who took more initiative or who had more control over what happened. So I've experimented with training a model to fill in for the referee! So far, it has an accuracy of ~60% on the test set, up from the 33% that random chance would give (choosing between left, right and no point eitherway). 

There are a few big challenges here, the primary one being that there is no labelled dataset of fencing clips! In general, video classification is a less explored than image classification, and even some of the most popular video datasets (like SPORTS 1-M) can have very respectable accuracies achieved on them by just looking at a single frame. Most model archtectures used in video classification are a combination of convolutional and recurrent nets. Convolutional to extract the features from each frame, with a recurrent network interpereting these and arriving at a final decision. I think the most interesting proposal here is a fully recurrent convolutional network, where every layer is recurrent. For the moment I've tried an approach which needs less data to be successful, which is taking a pretrained InceptionV3 net and using that on each frame of every clip to extract the convolutional features (i.e converting each frame into a 1x2048 vector). In the inception network that feature vector is then interpreted by one fully connected layer for the Imagenet database but the same feature vector should work reasonably well as representations of what is happening in the frame. Then a multi-layer LSTM is trained on 'clips' of these feature vectors. 

So that the network picks up on the concept of relative motion faster, before converting each frame to a feature vector the dense optical flow of the frame with respect to the previous frame is computed, mapped to a colorwheel and overlaid on the frame. The original frame is conveted into black and white, so that the network doesn't have to learn to distinguish between movement and the original colors. The use of optical flow is inspired by Karen Simonyan and Andrew Zisserman in 'Two-Stream Convolutional Networks for Action Recognition in Videos'.

I hope people who are interested in exploring machine learning on new problems find the code here useful. It'll take you all the way through creating your own dataset to a decent example of how to load and use pretrained models, and train and serve your own model.

## Creating the fencing clips database

- Typing in progress

