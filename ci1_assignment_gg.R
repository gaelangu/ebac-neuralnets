## Neural Network Ensemble for the Diabetes Dataset
# Data import
diabetes = read.csv('./datasets/Diabetes.csv', header = F)
colnames(diabetes) = c('no.of.times.preg', 'plasma.glucose.conc', 'diastolic.pressure',
                       'triceps.skin.fold.thick', 'serum.insulin', 'bmi', 'diab.pedigree.func',
                       'age', 'positive.test')

# Factorizing class variable
diabetes$positive.test = as.factor(as.character(diabetes$positive.test))

summary(diabetes)

# ------------------------------------------------------------------------

# Train-test split
library(caTools)
set.seed(111)
diabetes$split = sample.split(diabetes$positive.test, SplitRatio = 0.7)

diabetes_train = subset(diabetes, split == T)
diabetes_test = subset(diabetes, split == F)

# Excluding split columns
diabetes_train = diabetes_train[-10]
diabetes_test = diabetes_test[-10]

# Writing CSV for train and test
write.table(diabetes_train, sep = ',', file = './datasets/Diabetes_train.csv', row.names = F)
write.table(diabetes_test, sep = ',', file = './datasets/Diabetes_test.csv', row.names = F)

# ------------------------------------------------------------------------

# Single-Layer Neural Network
# Building the neural net model
library(nnet)
diabetes_nnet = nnet(positive.test ~ ., data = diabetes_train,
                     size = 8, maxit = 10000, decay = 0.0001)

summary(diabetes_nnet)

# ------------------------------------------------------------------------

# Neural network plot
require(RCurl)

root.url<-'https://gist.githubusercontent.com/fawda123'
raw.fun<-paste(
  root.url,
  '5086859/raw/cc1544804d5027d82b70e74b83b3941cd2184354/nnet_plot_fun.r',
  sep='/'
)
script<-getURL(raw.fun, ssl.verifypeer = FALSE)
eval(parse(text = script))
rm('script','raw.fun')

plot(diabetes_nnet)

# ------------------------------------------------------------------------

# Prediction and confusion matrix
test_pred = predict(diabetes_nnet, newdata = diabetes_test, type = 'class')
library(caret)
cm_nn = confusionMatrix(table(true = diabetes_test$positive.test, predicted = test_pred))
cm_nn

# ------------------------------------------------------------------------

# Neural Networks with Feature Extraction
# Building the neural net model
diabetes_pcann = pcaNNet(positive.test ~ ., data = diabetes_train,
                         size = 8, maxit = 10000, decay = 0.0001)
summary(diabetes_pcann)

# ------------------------------------------------------------------------

# Prediction and confusion matrix
test_pred_pcann = predict(diabetes_pcann, newdata = diabetes_test, type = 'class')

cm_pcann = confusionMatrix(table(true = diabetes_test$positive.test, predicted = test_pred_pcann))
cm_pcann

# ------------------------------------------------------------------------

# Ensemble Learning
# Coercing class variable into syntactically valid names
diabetes_train$positive.test = make.names(diabetes_train$positive.test)

library(caretEnsemble)
set.seed(111)
control1 = trainControl(method = 'repeatedcv', number = 10, repeats = 3, 
                        savePredictions = 'all', classProbs = TRUE)
algorithmList1 = c('glm', 'nnet', 'pcaNNet')

# Model Fit
models1 = caretList(positive.test ~ ., data = diabetes_train, 
                    trControl = control1, methodList = algorithmList1)

# Result summary
result1 = resamples(models1)
summary(result1)
dotplot(result1)

stack_control1 = trainControl(method = 'repeatedcv', number = 10, repeats = 3)
stack_glm1 = caretStack(models1, method = 'glm', metric = 'Accuracy', trControl = stack_control1)
print(stack_glm1)

# ------------------------------------------------------------------------

## Neural Network Emsemble for Wine Quality Dataset
# Data import
wine = read.csv('./datasets/winequality-white.csv')

head(wine)

# ------------------------------------------------------------------------

# Train-test split
set.seed(111)
wine$split = sample.split(wine$quality, SplitRatio = 0.7)

wine_train = subset(wine, split == T)
wine_test = subset(wine, split == F)

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

wine_train <- as.data.frame(lapply(wine_train, normalize))
wine_test <- as.data.frame(lapply(wine_test, normalize))

# Excluding split columns
wine_train = wine_train[-13]
wine_test = wine_test[-13]

# Writing CSV for train and test
write.table(wine_train, sep = ',', file = './datasets/winequality_train.csv', row.names = F)
write.table(wine_test, sep = ',', file = './datasets/winequality_test.csv', row.names = F)

# ------------------------------------------------------------------------

# SVM with Radial Basis Function Kernel
# Building the neural network model
library(kernlab)
wine_svmrbf = ksvm(quality ~ .,
                   data = wine_train,
                   kernel = 'rbfdot',
                   kpar = 'automatic')

summary(wine_svmrbf)

# ------------------------------------------------------------------------

# Prediction
rbf_pred = predict(wine_svmrbf, newdata = wine_test[-12])

model_values1 = data.frame(obs = as.numeric(wine_test$quality), pred = rbf_pred)
defaultSummary(model_values1)

# ------------------------------------------------------------------------

# Stacked AutoEncoder Deep Neural Network
# Building the neural network model
library(deepnet)
wine_dnn = dbn.dnn.train(x = as.matrix(wine_train[1:11]),
                         y = as.matrix(wine_train[12]),
                         learningrate = 0.001,
                         hidden = 8,
                         numepochs = 10)

summary(wine_dnn)

# ------------------------------------------------------------------------

# Prediction
wine_pred_dnn = nn.predict(wine_dnn, x = as.matrix(wine_test[1:11]))

model_values2 = data.frame(obs = as.numeric(wine_test$quality), pred = wine_pred_dnn)
defaultSummary(model_values2)

## ------------------------------------------------------------------------

# Ensemble Learning
library(caret)
library(caretEnsemble)
set.seed(111)

control2 <- trainControl(
  method="boot",
  number=25,
  savePredictions="final",
  classProbs=TRUE,
  index=createResample(wine_train$quality, 25))

models2 <- caretList(
  quality~., data=wine_train,
  trControl=control2,
  methodList=c("svmRadial", "dnn"))

greedy_ensemble <- caretEnsemble(
  models2,
  trControl=control2)

summary(greedy_ensemble)
result2 <- resamples(models2)

# Result summary
summary(result2)
dotplot(result2)

# ------------------------------------------------------------------------

set.seed(111)
stack_control2 = trainControl(method = 'boot', number = 25)
stack_glm2 = caretStack(models2, method = 'glm', metric = 'RMSE', trControl = stack_control2)
print(stack_glm2)

# ------------------------------------------------------------------------