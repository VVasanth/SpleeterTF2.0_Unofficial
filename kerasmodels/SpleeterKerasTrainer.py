import kerasmodels.SpleeterTrainKerasBase as SpleeterTrainKerasBase

noOfEpochs = 100
saveModelEvery = 10
startEpochVal = 0
modelPath = None
learningRate = 0

SpleeterTrainKerasBase.trainModelOverEpochs(noOfEpochs, saveModelEvery,startEpochVal, modelPath, learningRate)