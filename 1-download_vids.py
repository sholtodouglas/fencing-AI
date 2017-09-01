from pytube import YouTube
import os

## Run this to download all the videos from youtube.

import signal
import time
import traceback
## Timeout for use with try/except so that pytube doesn't randomly freeze.
class Timeout():
    """Timeout class using ALARM signal."""
    class Timeout(Exception):
        pass
 
    def __init__(self, sec):
        self.sec = sec
 
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)
 
    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm
 
    def raise_timeout(self, *args):
        raise Timeout.Timeout()



# Create all the directories needed if they don't yet exist
# The only non automatic step is right before optical flow production. Some data will be in 'training quarantine', some in 'more_training data'
# (i.e, the extra augmented data). Up to the user to copy these over to final training_clips when ready. 
directories = ['precut','videos','training_quarantine','more_training_data','final_training_clips','optical_flow','preinception_data', 'final_training_data', 'training_data']
for dirs in directories:
	if not os.path.exists(dirs):
	    os.makedirs(dirs)



text_file = open("sabre_videos.txt", "r")
vids = text_file.read().split('\r')
print "First 3 links:", vids[:3]
text_file.close()

# Loop through all the videos, download them and put them in the precut folder.
counter = 0
vids = vids[counter:]
for i in vids:

	
	try:
		with Timeout(600):
			start = time.time()
			yt = YouTube(i)
			yt.set_filename(str(counter))
			video = yt.get('mp4', '360p')
			video.download(os.getcwd()+ '/precut/')
			
			print "Downloaded: ", i, "   " ,(time.time() - start), "s"
	except:
		traceback.print_exc()
		print "Failed-",i



	counter = counter + 1




