This is an unofficial repository that contains the Spleeter from Deezer(https://github.com/deezer/spleeter/tree/master) migrated into TF2.0.

If you would like to participate over Gitter chat - pls use this channel: https://gitter.im/audioSourceSeparationOnEdge/community

Spleeter is an industry standard audio source separation library that exhibits amazing performance in splitting the audio files into various stems such as vocals, piano, drums, bass
and accompainments. Spleeter has been created in TF1.0 and the models are available in checkpoint format.

Building mobile compatible TFLite models from checkpoint files are very difficult and there are multiple requests to migrate Spleeter into TF2.0 to the official community.

This project is my effort in migrating Spleeter into TF2.0 leveraging the latest features of Tensorflow.

Library is currently being built and pls expect considerable changes over time.

Solution has been trained on very limited set of data and the built model achieves decent accuracy on the audio separation tasks from train/test data. I am working towards
continuously improving the model accuracy - but the current version is good to showcase the functioning of the model.

I would work towards training the solution on the complete dataset of musdb with appropriate infrastructure. Pls expect the very good version of this solution soon.

*******************************************************************************

Pls note that this project is an e2e solution - where the predicted spectrogram values are further processed to generate the audio files with appropriate masks.

'SpleeterTrainer.py' file is primarily associated with the training of model, whereas 'SpleeterValidator.py' file would perform following activities:

--> read the raw audio file,
--> process it into spectrogram
--> feed the spectrogram to the model to generate the predicted (single stem) spectrogram
--> process the predicted spectrogram to generate the functioning audio file.

As the model is providing lower accuracy now, in order to validate the rest of the processing - I have saved the predictions from Official Spleeter as an 'npy' file and bundled it into this project.
To validate how the processing works, pls run 'SpleeterValidator.py' file with the flag 'sampleRunToValidateProcessingFlag' as 'True'. When running with this flag, file would replace the predictions value with 
the prediction value of official spleeter and generate the audio file. You can try this and observe the transformation process.

*******************************************************************************

As stated above, objective of this project is to build an Android app with an on-device audio source separation capability. This project is primarily concerned with the model building activity.
Pls refer the below project for the progress on Android project, that utilizes the model generated from this project.

https://github.com/VVasanth/Spleeter_Unofficial_TF20_MobileApp

*******************************************************************************

Model building activity would require GPU and currently, we are leveraging the GPU from Google Collab. Google Collab files have been built for this purpose and they have been committed under the
'GoogleCollab' directory.

*******************************************************************************

**Environment Setup:**

Conda Env Setup:

1. Ensure you have anaconda installed on your machine.
2. Create a new environment with the below command:  
    conda create --name <envName> python=3.6
3. Once the installation is done, activate the environment with below command:
    conda activate <envName>
4. Install the dependency packages with the below command:
    pip install -r requirements.txt 


Input Data Set:
1. Input data set for this solution is musdb dataset.
2. MusDB dataset is a protected dataset and the request for the access has to be placed in the below link:
    https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems
3. Once request is placed, dataset access would be provided by the owners in a period of 24 hours.
4. Dataset would be around 4.4 GB in size and there will be two folders 'train' and 'test' with 100 and 50 files respectively.
5. You would observe the dataset to be in the format of '.stem.mp4'
6. We would be required to decode the '.stem.mp4' file into respective stems for us to feed the data for training.
7. Place the dataset under 'musdb_dataset' folder:
    ```shell script
    mkdir musdb_dataset && 
       cd musdb_dataset && 
       unzip ~/Downloads/musdb18.zip && 
       mv musdb18/t* . && cd ..
    ```       
8. Execute `. musdecode.sh` shell script and it will decode each of the files, extract the stems and place them under the respective directory.
9. After it ran folders would contain individual stems of the respective musics such as:

    a) bass.wav 
    b) drums.wav
    c) mixture.wav
    d) other.wav
    e) vocals.wav


 **Configurations**
 
 Files under 'config' folders are used for configurations. 
 
 1. 'musdb_config.json' is the main config file that contains various config parameters requried for the execution.
 2. 'musdb_train.csv' and 'musdb_validation.csv' are the files that are the config files from official Spleeter.
 3. 'musdb_train_small.csv' and 'musdb_validation_small.csv' are the files that contains smaller data - as I would be running the training in my CPU bound machine.    

**Training Operation**

Once you have extracted the input data from musdb dataset and configured the config files properly - you can start the training process from 'SpleeterTrain.py' file.

SpleeterTrain.py file has two components, as below:

--> Data processing: Data processing is primarily reads the input data from the csv files, loads the audio files and generates the stft values for the audio files. 
It does series of filtering to remove the invalid data. Data processing is almost same as official Spleeter and the code has been migrated to TF2.0.

--> Model Building: Official Spleeter generates the model as checkpoint files and generating TFLite model from this was difficult - thats the primary reason for building this 
version. Unet architecture of the network available in 'model/KerasUnet.py' is exactly similar to Official spleeter. But the train operation is performed using Tensorflow GradientTape function.

Let me summarize the process of ModelBuilding and the Unet architecture:

--> Spleeter primarily deals with audio separation process and the Unet architecture that is used here - primarily designed for Image Segmentation problems.  
--> Here, we will be using the same network and train the network by inputing 'mixture.wav' against each of the stems.  
--> Train operation would be performed for each of the stems individually and right now, model has been built with the 'vocals' separation data.
--> Training the model for rest of the stems would be straightforward - where you would be required to udpate the '_instruments' variable in 'SpleeterTrainer.py' file.
--> We will identify loss function as a measure of difference between the stems actual value and output value.  
--> Optimizer's function is to reduce the loss as part of the training process.  
--> Tensorflow 2 Keras api are used to build the training process.
--> Training script is written in such a way that it supports 'start/stop/resume' training operation.
--> During training process, models will be saved every 'n' epoch (as per the defined value in 'SpleeterTrainer.py' file) under 'kerasmodels/models' folder.
--> Training and validation losses are recorded as a plot under the directory 'kerasmodels/models'.
--> Models will be saved in 'hdf5' format.
--> Post training, models in 'hdf5' format could be converted into 'savedmodel' format with the script 'kerasmodels/HDF5ToSavedModelGenerator.py'.
--> RMSE is being used as a measure of loss  
--> SavedModels will be required to be ported into TFLite version, for android app processing (with Spleeter_TFLiteModelGEnerator.py file). 

**Test Operation**

Test operation is about taking the built model and feeding it with the audio source. Output of the model would contain multiple files corresponding to the respective stems.

Audio file needs to be processed for the stft values and to be fed into the model. Output of the model needs to be masked, such that it could be generated as the audio file.

Place the input file that needs to be processed in 'input' folder and post execution, output will be generated under the 'output' folder.

Output of model produces decent extraction of vocal audio, which you can play in any player. 

**Accuracy Improvement Process**

Accuracy improvement is the activity that will be taken up in the coming weeks. Intent is to reduce the RMSE error to lesser value there by model will be able to separate the audio stems.
Till now, training operation has been performed on CPUs. GPU is required to train the model effectively across wide datasets across ~200K runs. 

Contributors interested towards improving the accuracy can get in touch with me.
