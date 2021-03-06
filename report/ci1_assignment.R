#' ---
#' title: "Computational Intelligence I Assignment"
#' author: "Gaelan Gu"
#' date: "20/04/2017"
#' output: word_document
#' ---
#' 
#' # Neural Network Ensemble for Diabetes Dataset
#' 
## ------------------------------------------------------------------------
# Data import
diabetes = read.csv('./datasets/Diabetes.csv', header = F)
colnames(diabetes) = c('no.of.times.preg', 'plasma.glucose.conc', 'diastolic.pressure',
                       'triceps.skin.fold.thick', 'serum.insulin', 'bmi', 'diab.pedigree.func',
                       'age', 'positive.test')

# Factorizing class variable
diabetes$positive.test = as.factor(as.character(diabetes$positive.test))

summary(diabetes)

#' 
## ------------------------------------------------------------------------
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

#' 
#' 
#' ## Single-Layer Neural Network
#' 
## ------------------------------------------------------------------------
# Building the neural net model
library(nnet)
diabetes_nnet = nnet(positive.test ~ ., data = diabetes_train,
                     size = 8, maxit = 10000, decay = 0.0001)

summary(diabetes_nnet)


#' 
## ------------------------------------------------------------------------
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

## ------------------------------------------------------------------------
plot(diabetes_nnet)

#' 
#' 
## ------------------------------------------------------------------------
# Prediction and confusion matrix
test_pred = predict(diabetes_nnet, newdata = diabetes_test, type = 'class')
library(caret)
cm_nn = confusionMatrix(table(true = diabetes_test$positive.test, predicted = test_pred))
cm_nn

#' 
#' Accuracy of **72.2%** achieved on test set with single-layer neural network.
#' 
#' 
#' ## Neural Networks with Feature Extraction
## ------------------------------------------------------------------------
diabetes_pcann = pcaNNet(positive.test ~ ., data = diabetes_train,
                         size = 8, maxit = 10000, decay = 0.0001)
summary(diabetes_pcann)

#' 
## ------------------------------------------------------------------------
# Prediction and confusion matrix
test_pred_pcann = predict(diabetes_pcann, newdata = diabetes_test, type = 'class')

cm_pcann = confusionMatrix(table(true = diabetes_test$positive.test, predicted = test_pred_pcann))
cm_pcann

#' 
#' 
#' ## Ensemble Learning
## ------------------------------------------------------------------------
set.seed(111)
control1 = trainControl(method = 'repeatedcv', number = 10, repeats = 3)

# NN Fits
fit.nn = train(positive.test ~ ., data = diabetes_train, method = 'nnet', metric = 'Accuracy', trControl = control1)
fit.pcann = train(positive.test ~ ., data = diabetes_train, method = 'pcaNNet', metric = 'Accuracy', trControl = control1)

# Result summary
result1 = resamples(list(nn = fit.nn, pcann = fit.pcann))
summary(result1)
dotplot(result1)

#' 
#' 
#' PCANN performs better at a mean accuracy of **77.1%**.
#' 
#' 
#' # Neural Network Emsemble for Wine Quality Dataset
#' 
## ------------------------------------------------------------------------
# Data import
wine = read.csv('./datasets/winequality-white.csv')

head(wine)

#' 
## ------------------------------------------------------------------------
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

#' 
## ------------------------------------------------------------------------
wine_nnet = nnet(quality ~ ., data = wine_train,
                     size = 10, maxit = 1000, decay = 0.0001, linout = T)

plot(wine_nnet)

#' 
## ------------------------------------------------------------------------
wine_pred_nnet = predict(wine_nnet, newdata = wine_test)
wine_pred_nnet = as.numeric(round(wine_pred_nnet))

model_values0 = data.frame(obs = as.numeric(wine_test$quality), pred = wine_pred_nnet)
defaultSummary(model_values0)

#' 
#' 
#' ## Radial Basis Function (RBF) Neural Network
## ------------------------------------------------------------------------
# Building the RBF
library(RSNNS)
wine_rbfnet = rbf(x = wine_train[-12],
                  y = wine_train[12],
                  size = c(8),
                  maxit = 1000,
                  linOut = T)

summary(wine_rbfnet)

#' 
## ------------------------------------------------------------------------
# Prediction
rbf_pred = predict(wine_rbfnet, newdata = wine_test[-12])
rbf_pred = as.numeric(round(rbf_pred))

model_values1 = data.frame(obs = as.numeric(wine_test$quality), pred = rbf_pred)
defaultSummary(model_values1)

#' 
#' ## Bayesian Regularized Neural Network
## ------------------------------------------------------------------------
library(brnn)
wine_brnn = brnn(quality ~ ., data = wine_train,
                 neurons = 8,
                 epochs = 1000)
wine_brnn

#' 
## ------------------------------------------------------------------------
# Prediction
wine_pred_brnn = predict(wine_brnn, newdata = wine_test)
wine_pred_brnn = as.numeric(round(wine_pred_brnn))

# l = sort(union(wine_pred_brnn, wine_test$quality))
# wine_pred_brnn = factor(as.character(wine_pred_brnn), levels = l)

# Confusion Matrix
# cm_table = table(wine_pred_brnn, factor(as.character(wine_test$quality), levels = l))
# cm_brnn = confusionMatrix(cm_table)
# cm_brnn

model_values2 = data.frame(obs = as.numeric(wine_test$quality), pred = wine_pred_brnn)
defaultSummary(model_values2)

#' 
#' ## Ensemble Learning
#' 
## ------------------------------------------------------------------------
library(caret)
library(caretEnsemble)
set.seed(111)
control1 = trainControl(method = 'cv', number = 10, repeats = 3,savePredictions=TRUE)


# NN Fits
#fit.nn = train(quality~ ., data = wine_train, method = 'rbf', trControl = control1)
#fit.pcann = train(quality ~ ., data = wine_train, method = 'brnn', metric = 'Accuracy', trControl = control1)
algorithmList <- c( 'rbfDDA','DENFIS') #, 'brnn'
#models <- caretList(quality~., data=wine_train, trControl=control1, methodList=algorithmList)

control2 <- trainControl(
  method="boot",
  number=25,
  savePredictions="final",
  classProbs=TRUE,
  index=createResample(wine_train$quality, 25)
)

model_list <- caretList(
  quality~., data=wine_train,
  trControl=control2,
  methodList=c("svmRadial", "dnn")
)
#library("mlbench")
#library("pROC")
#data(Sonar)

greedy_ensemble <- caretEnsemble(
  model_list,
  trControl=control2)

summary(greedy_ensemble)
result1 <- resamples(model_list)

# Result summary
#result1 = resamples(list(nn = fit.nn, pcann = fit.pcann))
summary(result1)
dotplot(result1)


model_preds <- lapply(model_list, predict, newdata=wine_test[-12])
enseble_predict <-round (predict(greedy_ensemble,wine_test[-12] ))






#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' 
