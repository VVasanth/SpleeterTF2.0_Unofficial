This is an unofficial repository that contains the Spleeter from Deezer(https://github.com/deezer/spleeter/tree/master) migrated into TF2.0.

Spleeter is an industry standard audio source separation library that exhibits amazing performance in splitting the audio files into various stems such as vocals, piano, drums, bass
and accompainments. Spleeter has been created in TF1.0 and the models are available in checkpoint format.

Building mobile compatible TFLite models from checkpoint files are very difficult and there are multiple requests to migrate Spleeter into TF2.0 to the official community.

This project is my effort in migrating Spleeter into TF2.0 leveraging the latest features of Tensorflow.

Library is currently being built and pls expect considerable changes over time.

Solution has been trained on very limited set of data and does not exhibit proper accuracy in audio separation tasks. But the underlying training and testing code has been faithfully 
migrated referring the source code of official Spleeter.

I would work towards training the solution on the complete dataset of musdb with appropriate infrastructure. Pls expect the functioning version of this solution soon.
