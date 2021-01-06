This is an unofficial repository that contains the Spleeter from Deezer(https://github.com/deezer/spleeter/tree/master) migrated into TF2.0.

If you would like to participate over Gitter chat - pls use this channel: https://gitter.im/audioSourceSeparationOnEdge/community

Spleeter is an industry standard audio source separation library that exhibits amazing performance in splitting the audio files into various stems such as vocals, piano, drums, bass
and accompainments. Spleeter has been created in TF1.0 and the models are available in checkpoint format.

Building mobile compatible TFLite models from checkpoint files are very difficult and there are multiple requests to migrate Spleeter into TF2.0 to the official community.

This project is my effort in migrating Spleeter into TF2.0 leveraging the latest features of Tensorflow.

Library is currently being built and pls expect considerable changes over time.

Solution has been trained on very limited set of data and does not exhibit proper accuracy in audio separation tasks. But the underlying training and testing code has been faithfully 
migrated referring the source code of official Spleeter.

I would work towards training the solution on the complete dataset of musdb with appropriate infrastructure. Pls expect the functioning version of this solution soon.

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
--> We will identify loss function as a measure of sum of difference between each of the stems actual value and output value.  
--> Optimizer's funciton is to reduce the summation of loss as part of the training process.  
--> As we are required to consier the summation of each of the stem's loss and optimize it - keras based model building would not work here.  
--> We need to write 'custom training logic' using tensorflow's gradient tape's feature.  
--> SpleeterTrain.py contains the corresponding code and trains the network.  
--> Losses of the network are measured every 5 runs and model is saved every 10 runs. These are configurable values and we can change them, as required.  
--> While saving the model - we are saving both checkpoint and savedmodel version.  
--> SavedModel version is to generate the TFLite model, whereas checkpoint is to restart the training post the completion of one run.  
--> Models will be available in 'spleeter_saved_model_dir' at the end of the training.  
--> RMSE is being used as a measure of loss  
--> When running the training loop across ~2000 runs on a limited dataset - loss is around 3.0  
--> As part of training operation - 4 model files will be generated, each corresponding to each of the stems.  
--> 4 models will be required to be ported into TFLite version, for android app processing  

**Test Operation**

Test operation is about taking the built model and feeding it with the audio source. Output of the model would contain multiple files corresponding to the respective stems.

Audio file needs to be processed for the stft values and to be fed into the model. Output of the model needs to be masked, such that it could be generated as the audio file.

Place the input file that needs to be processed in 'input' folder and post execution, output will be generated under the 'output' folder.

Output of model produces the distorted audio, which could be played in any player. 

**Accuracy Improvement Process**

Accuracy improvement is the activity that will be taken up in the coming weeks. Intent is to reduce the RMSE error to lesser value there by model will be able to separate the audio stems.
Till now, training operation has been performed on CPUs. GPU is required to train the model effectively across wide datasets across ~200K runs. 

Contributors interested towards improving the accuracy can get in touch with me.
