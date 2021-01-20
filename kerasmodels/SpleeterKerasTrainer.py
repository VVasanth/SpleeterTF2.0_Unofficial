import kerasmodels.SpleeterTrainKerasBase as SpleeterTrainKerasBase

noOfEpochs = 5
saveModelEvery = 3
startEpochVal = 0
modelPath = None
learningRate = 0

SpleeterTrainKerasBase.trainModelOverEpochs(noOfEpochs, saveModelEvery,startEpochVal, modelPath, learningRate)