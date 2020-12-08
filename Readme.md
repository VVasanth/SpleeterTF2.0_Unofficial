This is an unofficial repository that contains the Spleeter from Deezer(https://github.com/deezer/spleeter/tree/master) migrated into TF2.0.

Spleeter is an industry standard audio source separation library that exhibits amazing performance in splitting the audio files into various stems such as vocals, piano, drums, bass
and accompainments. Spleeter has been created in TF1.0 and the models are available in checkpoint format.

Building mobile compatible TFLite models from checkpoint files are very difficult and there are multiple requests to migrate Spleeter into TF2.0 to the official community.

This project is my effort in migrating Spleeter into TF2.0 leveraging the latest features of Tensorflow.

Library is currently being built and pls expect considerable changes over time.

Solution has been trained on very limited set of data and does not exhibit proper accuracy in audio separation tasks. But the underlying training and testing code has been faithfully 
migrated referring the source code of official Spleeter.

I would work towards training the solution on the complete dataset of musdb with appropriate infrastructure. Pls expect the functioning version of this solution soon.

**Environment Setup:**

Input Data Set:
1. Input data set for this solution is musdb dataset.
2. MusDB dataset is a protected dataset and the request for the access has to be placed in the below link:
    https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems
3. Once request is placed, dataset access would be provided by the owners in a period of 24 hours.
4. Dataset would be around 4.4 GB in size and there will be two folders 'train' and 'test' with 100 and 50 files respectively.
5. You would observe the dataset to be in the format of '.stem.mp4' and each of the files would contain individual stems of the respective musics such as:

    a) bass.wav 
    b) drums.wav
    c) mixture.wav
    d) other.wav
    e) vocals.wav
6) We would be required to decode the '.stem.mp4' file into respective stems for us to feed the data for training.
7) Place the dataset under 'musdb_dataset' folder.
8) Execute the 'musdecode.sh' shell script and it will decode each of the files, extract the stems and place them under the respective directory.

 **Configurations**
 
 Files under 'config' folders are used for configurations. 
 
 1. 'musdb_config.json' is the main config file that contains various config parameters requried for the execution.
 2. 'musdb_train.csv' and 'musdb_validation.csv' are the files that are the config files from official Spleeter.
 3. 'musdb_train_small.csv' and 'musdb_validation_small.csv' are the files that contains smaller data - as I would be running the training in my CPU bound machine.    

