import wave
import numpy as np
from struct import *
import matplotlib.pyplot as plt

audio_file = "test_audio.wav"
audio_read = wave.open(audio_file, "r")

whole_file = []
while audio_read.tell() < audio_read.getnframes():
	decoded = int(unpack("<h", audio_read.readframes(1))[0])
	whole_file.append(decoded)

# normalize just to visualize it easier
whole_file = np.array(whole_file).astype(np.float)
max_val = 32768.0
min_val = 32767.0

# normalizing the whole range of values
negatives = np.where(whole_file < 0)
whole_file[negatives] = whole_file[negatives] / min_val

positives = np.where(whole_file >= 0)
whole_file[positives] = whole_file[positives] / max_val


plt.plot(whole_file)
plt.ylabel('Waveform')
plt.show()


# noise
for idx in range(3):
	factor = 0.3*idx
	noised = whole_file*(1-factor) + (np.random.random(whole_file.shape)-np.random.random(whole_file.shape))*factor
	plt.plot(noised)
	plt.ylabel('Waveform with noise %f' % factor)
	plt.show()



# now how do we send it to tensorflow?
import tensorflow as tf
TFwav = tf.placeholder("float", shape=[whole_file.shape[0]])

with tf.Session() as sess:
	# eval expressions with parameters for the image
	wave_got = sess.run(TFwav, feed_dict={TFwav: noised})
	plt.plot(wave_got)
	plt.ylabel('Waveform got from Tensorflow')
	plt.show()
