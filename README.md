# Using Deep Learning to Referee Fencing

In fencing, when both competitors hit eachother within a few miliseconds, both sides of the scoreboard will light up. Then the referee has to make a decision on who's point it is. As a rough rule of thumb, it goes to who took more initiative or who had more control over what happened. So I've experimented with training a model to fill in for the referee! So far, it has an accuracy of ~60% on the test set, up from the 33% that random chance would give (choosing between left, right and no point eitherway). 

There are a few big challenges here, the primary one being that there is no labelled dataset of fencing clips!
- Typing in process

