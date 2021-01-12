import kerasmodels.SpleeterTrainKerasBase as SpleeterTrainKerasBase

noOfEpochs = 5
saveModelEvery = 3
startEpochVal = 0
modelPath = '../kerasmodels/models/model_50.hdf5'
learningRate = 1e-4

SpleeterTrainKerasBase.trainModelOverEpochs(noOfEpochs, saveModelEvery,startEpochVal, modelPath, learningRate)