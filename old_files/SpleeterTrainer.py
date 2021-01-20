from old_files import SpleeterTrainBase

noOfTrainingRuns = 101
saveModelAtEveryRun = 50
#models will be saved under the directory 'spleeter_saved_model_dir'
SpleeterTrainBase.trainModel(noOfTrainingRuns, saveModelAtEveryRun)