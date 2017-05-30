---
output:
  word_document: default
  html_document: default
---

## CA Assignment:
#### Neural Network Ensembles
----

##### Given :  Two benchmark classification/regression problems:
-  Diabetes.csv 

The diabetes data set contains the diagnostic data to investigate whether the
patient shows signs of diabetes according to World Health Organization criteria
such as the 2-hour post-load plasma glucose.

-  Winequality-white.csv


The winequality-white data is related to the white variants of the Portuguese
"Vinho Verde" wine. The goal is to model wine quality based on physicochemical
tests.

##### Expected :
1. Train a group of different types of NNs using different NN tools to solve the two problems given. 
(Use 2 different tools to train 2-3 different types of NNs)
2. Work on the two data sets
 You may partition each data set into two subsets: eg 75% as training data and
25% as test data
3. Train the NNs to achieve the highest possible classification accuracy or lowest
possible MSE.
4. NN ensemble - combine the outputs of individual NNs for final output 
(you may define certain calculation, such as rule(s) for the integration)
Compare the NN performance between the NN ensemble and the individual NNs



```python
#all imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from neupy import algorithms, estimators, environment,layers
from sklearn.metrics import confusion_matrix
%matplotlib inline
```

 ## 1. [ Diabities Problem ]

For Each Attribute: (all numeric-valued)
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)
   
   
Class Distribution: (class value 1 is interpreted as "tested positive for
   diabetes")

   Class Value  Number of instances
   0            500
   1            268


```python
df_diab = pd.read_csv('Diabetes.csv')
df_diab.columns = ['nop',
                   'pgc',
                   'bp',
                   'sft',
                   'sein',
                   'bmi',
                   'pedig',
                   'age',
                   'cls'
                  ]
df_diab.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nop</th>
      <th>pgc</th>
      <th>bp</th>
      <th>sft</th>
      <th>sein</th>
      <th>bmi</th>
      <th>pedig</th>
      <th>age</th>
      <th>cls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>116</td>
      <td>74</td>
      <td>0</td>
      <td>0</td>
      <td>25.6</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#total
print("total size of records")
np.size(df_diab)
```

    total size of records





    6903




```python
X1 = df_diab.iloc[:,:8]
y1 = df_diab['cls']
X1,y1 = shuffle(X1,y1)
#X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(preprocessing.minmax_scale(X1),preprocessing.minmax_scale(y1),train_size=0.70)
```

    /usr/local/anaconda3/envs/carnd-term1/lib/python3.5/site-packages/sklearn/utils/validation.py:429: DataConversionWarning: Data with input dtype int64 was converted to float64.
      warnings.warn(msg, _DataConversionWarning)



```python
print("train",X1_train.shape)
print("test",X1_test.shape)
print("train_y",y1_train.shape)
```

    train (536, 8)
    test (231, 8)
    train_y (536,)



```python

print(y1_train[0])
X1_train[0]
```

    0.0





    array([ 0.05882353,  0.53768844,  0.55737705,  0.19191919,  0.        ,
            0.39493294,  0.03714774,  0.05      ])




```python

```


```python

```


```python
X1_test[0]
```




    array([ 0.35294118,  0.52763819,  0.57377049,  0.32323232,  0.08037825,
            0.45901639,  0.01878736,  0.26666667])




```python
theta = 2 # readius
epsilon = 1e-4
(size,nf) = X1_train.shape
# activation function
def rce_activation(X,weights):
    z = np.dot(X,weights) #distance matrix for d(X,Wi)
    print("z is",X.shape, weights.shape, z.shape)
    f = 1 if z <= theta else 0 #threshold
    return(f)


# model
def rce_network_train(X):
    weights = np.array(X)
    biases = np.zeros((size,nf))
    input_layer = np.matmul(weights.transpose(),X)
    
    #y = rce_activation(X,input_layer)
    #print('y is ' + y)
    
    
    print(input_layer.shape)
    
    #lamdba = np.zeros((1,nf))
    
    #for i in range(1,)
    
    return(input_layer)

    
    
```

 ## 1a.  GRNN Network - [ Diabities Problem ]


```python
grnn_nw = algorithms.GRNN(std=0.1, verbose=True)
print(grnn_nw)

```

    
    Main information
    
    [ALGORITHM] GRNN
    
    [OPTION] verbose = True
    [OPTION] epoch_end_signal = None
    [OPTION] show_epoch = 1
    [OPTION] shuffle_data = False
    [OPTION] step = 0.1
    [OPTION] train_end_signal = None
    [OPTION] std = 0.1
    
    GRNN(std=0.1, show_epoch=None, train_end_signal=None, shuffle_data=None, verbose=True, epoch_end_signal=None, step=None)



```python
grnn_nw.train(X1_train, y1_train)
```


```python
y1_predicted = grnn_nw.predict(X1_test).round()

y1_predicted[0]
```




    array([ 0.])




```python
#accuracy
estimators.rmse(y1_predicted, y1_test)
```




    0.5425608669746597




```python
#confusion matrix
confusion_matrix(y1_test,y1_predicted)
```




    array([[128,  26],
           [ 42,  35]])




```python
from sklearn.metrics import accuracy_score
grnn_acc_score = accuracy_score(y1_test, y1_predicted)
print("Grnn accuracy score ", grnn_acc_score)
```

    Grnn accuracy score  0.705627705628


 ## 1b.  PNN Network - [ Diabities Problem ]



```python
pnn_nw = algorithms.PNN(std=10, verbose=False)
print(pnn_nw)
```

    PNN(std=10, show_epoch=1, train_end_signal=None, shuffle_data=False, verbose=False, epoch_end_signal=None, batch_size=128, step=0.1)



```python
pnn_nw.train(X1_train, y1_train)
```


```python
y1_pnn_predicted = pnn_nw.predict(X1_test).round()
y1_pnn_predicted[0]
```




    0.0




```python
#accuracy
estimators.rmse(y1_pnn_predicted, y1_test)
```




    0.5385566730097122




```python
#confusion matrix
confusion_matrix(y1_test,y1_pnn_predicted)
```




    array([[130,  24],
           [ 43,  34]])




```python
pnn_acc_score = accuracy_score(y1_test, y1_pnn_predicted)
print("Pnn accuracy score ", pnn_acc_score)
```

    Pnn accuracy score  0.709956709957


 ## 1c.  RBF - [ Diabities Problem ]




```python
rbf_nw = algorithms.RBFKMeans(n_clusters=2, verbose=False)
```


```python
rbf_nw.train(X1_train, epsilon=1e-5)
```


```python
y1_rbf_predicted = rbf_nw.predict(X1_test)
```


```python
confusion_matrix(y1_test,y1_rbf_predicted)
```




    array([[117,  37],
           [ 45,  32]])




```python
rbf_acc_score = accuracy_score(y1_test, y1_rbf_predicted)
print("RBF accuracy score ", rbf_acc_score)
```

    RBF accuracy score  0.645021645022


## 1d. Ensemble Learning - [ Diabities Problem ]


```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = SVC(kernel='rbf', probability=True)
mlp_nw = MLPClassifier(solver='lbfgs', alpha=0.01,max_iter=2000,  hidden_layer_sizes=(5, 2), random_state=1, activation='relu')
mv_clf = VotingClassifier(estimators=[('pnn_nw', pnn_nw), ('clf2', clf2), ('clf3', clf3),('clf4', clf4),('mlp_nw', mlp_nw)], voting='hard')
#mv_clf = MajorityVoteClassifier(classifiers=[grnn_nw,pnn_nw,rbf_nw])
```


```python
mv_clf = mv_clf.fit(X1_train, y1_train)
```


```python
X1_test.shape
#X1_train.shape
#mv_clf
#y_mv_clf_predicted = mv_clf.predict(X1_test)
```




    (231, 8)




```python
y_mv_clf_predicted = mv_clf.predict(X1_test)
```


```python
y_mv_clf_predicted[0]
```




    0.0




```python
confusion_matrix(y1_test,y_mv_clf_predicted)
```




    array([[139,  15],
           [ 37,  40]])




```python
mv_clf_acc_score = accuracy_score(y1_test, y_mv_clf_predicted)
print("Ensembles accuracy score ", mv_clf_acc_score)
```

    Ensembles accuracy score  0.774891774892


 ## 1e.  MLP - [ Diabities Problem ]




```python
from sklearn.neural_network import MLPClassifier
mlp_nw = MLPClassifier(solver='lbfgs', alpha=0.01,max_iter=2000,  hidden_layer_sizes=(5, 2), random_state=1, activation='relu')
#sgd
```


```python
mlp_model = mlp_nw.fit(X1_train, y1_train)
mlp_model
```




    MLPClassifier(activation='relu', alpha=0.01, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(5, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=2000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)




```python
y1_mlp_predicted = mlp_model.predict(X1_test)
```


```python
y1_mlp_predicted[1]
```




    0.0




```python

confusion_matrix(y1_test,y1_mlp_predicted)
```




    array([[152,   2],
           [ 62,  15]])




```python
mlp_acc_score = accuracy_score(y1_test, y1_mlp_predicted)
print("Pnn accuracy score ", mlp_acc_score)
```

    Pnn accuracy score  0.722943722944



```python

```

 ## 1f.  MLFF with tensorflow - [ Diabities Problem ]




```python
#
# Parameters
learning_rate_1 = 0.001
training_epochs_1 = 20000
batch_size_1 = 10
display_step_1 = 1000

# Network Parameters
n_hidden_1 = 8 # 1st layer number of features
n_hidden_2 = 8 # 1st layer number of features
n_input = 8 # diabities data have 8 features and 1 output with 2 classes
n_classes = 2 # 2 classess 
```


```python
# tf Graph input
x = tf.placeholder("float", [None, n_input],name="x")
y = tf.placeholder("float", [None,n_classes],name="y")
```


```python
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
```


```python
from sklearn import preprocessing
def one_hot(y_data) :
    enc = preprocessing.LabelEncoder()
    y_data_encoded = enc.fit_transform(y_data)
    #print(y_data_encoded)
    a = np.array(y_data_encoded, dtype=int)
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    #print(b)
    return b
```


```python
# Create model
def multilayer_perceptron_tf(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
```


```python
# Construct model
pred = multilayer_perceptron_tf(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_1).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
```


```python
errors = []
y1_train_h = one_hot(y1_train)
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

   
    # Training cycle
    for epoch in range(training_epochs_1):
        avg_cost = 0.
        #print(X1_train.shape, y1_train.shape)
        X1_train, y1_train_h = shuffle(X1_train,y1_train_h)
       # y1_train = pd.DataFrame(y1_train,columns=['cls'])
       # y1_train['e'] = pd.Series(0, index=y1_train.index)
        #print(X1_train.shape, y1_train.shape)
            # Run optimization op (backprop) and cost op (to get loss value)
        #print(y_train.shape)
        _, c = sess.run([optimizer, cost], feed_dict={x: X1_train, y: y1_train_h})
        
        # Display logs per epoch step
        if epoch % display_step_1 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(c))
            errors.append(c)
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(tf.round(pred), 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    y1_test_h = one_hot(y1_test)
    
    mlp_tf_acc_Score = accuracy.eval({x: X1_test, y: y1_test_h})
    print("Accuracy:", mlp_tf_acc_Score)
```

    Epoch: 0001 cost= 2.092828035
    Epoch: 1001 cost= 0.456179202
    Epoch: 2001 cost= 0.433209538
    Epoch: 3001 cost= 0.410290569
    Epoch: 4001 cost= 0.382849276
    Epoch: 5001 cost= 0.365616709
    Epoch: 6001 cost= 0.354704320
    Epoch: 7001 cost= 0.345349103
    Epoch: 8001 cost= 0.338299721
    Epoch: 9001 cost= 0.332837075
    Epoch: 10001 cost= 0.328014821
    Epoch: 11001 cost= 0.315553159
    Epoch: 12001 cost= 0.308456808
    Epoch: 13001 cost= 0.304505199
    Epoch: 14001 cost= 0.301903009
    Epoch: 15001 cost= 0.300433159
    Epoch: 16001 cost= 0.298976272
    Epoch: 17001 cost= 0.298128486
    Epoch: 18001 cost= 0.296585053
    Epoch: 19001 cost= 0.290240049
    Optimization Finished!
    Accuracy: 0.718615



```python
summary_1 = pd.DataFrame([[grnn_acc_score,pnn_acc_score,rbf_acc_score,mv_clf_acc_score,mlp_acc_score,mlp_tf_acc_Score]])
summary_1.columns=['GRNN', 'PNN', 'RBF', 'ENSEMBLE', 'MLP','MLP_TF']
```


```python
summary_1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRNN</th>
      <th>PNN</th>
      <th>RBF</th>
      <th>ENSEMBLE</th>
      <th>MLP</th>
      <th>MLP_TF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.705628</td>
      <td>0.709957</td>
      <td>0.645022</td>
      <td>0.774892</td>
      <td>0.722944</td>
      <td>0.718615</td>
    </tr>
  </tbody>
</table>
</div>



As seen above, the best classifier was with **Ensemble** learning.

## 2. Wine Quality

 Input variables (based on physicochemical tests):
   1. - fixed acidity
   2. - volatile acidity
   3. - citric acid
   4. - residual sugar
   5. - chlorides
   6. - free sulfur dioxide
   7. - total sulfur dioxide
   8. - density
   9. - pH
   10. - sulphates
   11. - alcohol
   Output variable (based on sensory data): 
   12. - quality (score between 0 and 10)


```python
df_wines = pd.read_csv('winequality-white.csv')
```


```python
df_wines.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>0.27</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.0010</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.3</td>
      <td>0.30</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.9940</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.1</td>
      <td>0.28</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.9951</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
X2 = df_wines.iloc[:,:11]
y2 = df_wines['quality']
X2,y2 = shuffle(X2,y2)
#X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(preprocessing.minmax_scale(X2),preprocessing.minmax_scale(y2),train_size=0.70)
```

    /usr/local/anaconda3/envs/carnd-term1/lib/python3.5/site-packages/sklearn/utils/validation.py:429: DataConversionWarning: Data with input dtype int64 was converted to float64.
      warnings.warn(msg, _DataConversionWarning)



```python
print("train",X2_train.shape)
print("test",X2_test.shape)
print("train_y",y2_train.shape)
```

    train (3428, 11)
    test (1470, 11)
    train_y (3428,)


### 2a. MLR ( Multi linear Regression)


```python
from sklearn.neural_network import MLPRegressor
mlr_nw = MLPRegressor(solver='lbfgs', alpha=0.01,max_iter=2000,  hidden_layer_sizes=(5, 2), random_state=1, activation='relu')
#sgd
```


```python
mlr_model = mlr_nw.fit(X2_train, y2_train)
mlr_model
```




    MLPRegressor(activation='relu', alpha=0.01, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(5, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=2000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)




```python
y2_mlr_predicted = mlr_model.predict(X2_test)
```


```python
y2_mlr_predicted[0]
```




    0.44020853688104605




```python
# The mean squared error
y2_mlr_mse = np.mean((y2_mlr_predicted - y2_test) ** 2)
print("Mean squared error: %.4f"
      % y2_mlr_mse)
```

    Mean squared error: 0.0154



```python
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % mlr_model.score(X2_test, y2_test))
```

    Variance score: 0.27



```python
# Plot outputs
#plt.scatter(X2_test[:,1:2], y2_test,  color='black')
#plt.plot(X2_test, y2_mlr_predicted, color='blue',
#         linewidth=3)

#plt.xticks(())
#plt.yticks(())
```

### 2b. GRNN


```python

```


```python
grnn_nw_2 = algorithms.GRNN(std=0.1, verbose=True)
print(grnn_nw_2)
```

    
    Main information
    
    [ALGORITHM] GRNN
    
    [OPTION] verbose = True
    [OPTION] epoch_end_signal = None
    [OPTION] show_epoch = 1
    [OPTION] shuffle_data = False
    [OPTION] step = 0.1
    [OPTION] train_end_signal = None
    [OPTION] std = 0.1
    
    GRNN(std=0.1, show_epoch=None, train_end_signal=None, shuffle_data=None, verbose=True, epoch_end_signal=None, step=None)



```python
grnn_nw_2.train(X2_train, y2_train)
```


```python
y2_grnn_predicted = grnn_nw_2.predict(X2_test)
```


```python
y2_grnn_predicted[0]
```




    array([ 0.48209117])




```python
# The mean squared error
y2_grnn_mse = np.mean((y2_grnn_predicted - y2_test) ** 2)
print("Mean squared error: %.4f"
      % y2_grnn_mse)
```

    Mean squared error: 0.0294


### 2c. PNN Network 


```python
pnn_nw_2 = algorithms.PNN(std=10, verbose=False)
print(pnn_nw_2)
```

    PNN(std=10, show_epoch=1, train_end_signal=None, shuffle_data=False, verbose=False, epoch_end_signal=None, batch_size=128, step=0.1)



```python
pnn_nw_2.train(X2_train, y2_train)
```


```python
y2_pnn_predicted = pnn_nw_2.predict(X2_test)
```


```python
y2_pnn_predicted[0]
```




    0.33333333333333326




```python
# The mean squared error
y2_pnn_mse = np.mean((y2_pnn_predicted - y2_test) ** 2)
print("Mean squared error: %.4f"
      % y2_pnn_mse)
```

    Mean squared error: 0.0421


### 2d. RBF Network


```python
rbf_nw_2 = algorithms.RBFKMeans(n_clusters=2, verbose=False)
```


```python
rbf_nw_2.train(X2_train, epsilon=1e-5)
```


```python
y2_rbf_predicted = rbf_nw_2.predict(X2_test)
```


```python
y2_rbf_predicted[0]
```




    array([ 0.])




```python
 #The mean squared error
y2_rbf_mse = np.mean((y2_rbf_predicted - y2_test) ** 2)
print("Mean squared error: %.4f"
      % y2_rbf_mse)
```

    Mean squared error: 0.2690


:( Too high

### 2d. Ensembles


```python
from sklearn.ensemble import AdaBoostRegressor
#clf2 = RandomForestClassifier(random_state=1)
#clf3 = GaussianNB()
#clf4 = SVC(kernel='rbf', probability=True)

en_reg = AdaBoostRegressor(base_estimator=mlr_nw ,n_estimators=50)
#

```


```python
en_reg.fit(X2_train, y2_train)
```




    AdaBoostRegressor(base_estimator=MLPRegressor(activation='relu', alpha=0.01, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(5, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=2000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False),
             learning_rate=1.0, loss='linear', n_estimators=50,
             random_state=None)




```python
y2_ens_predicted = en_reg.predict(X2_test)
```


```python
y2_ens_predicted[0]
```




    0.45672889913636916




```python
# The mean squared error
y2_ens_mse = np.mean((y2_ens_predicted - y2_test) ** 2)
print("Mean squared error: %.4f"
      % y2_ens_mse)
```

    Mean squared error: 0.0148


### Summary


```python
summary_2 = pd.DataFrame([[y2_mlr_mse,y2_grnn_mse,y2_pnn_mse,y2_rbf_mse,y2_ens_mse]])
summary_2.columns=['MLR', 'GRNN', 'PNN', 'RBF','ENSEMBLE']
```


```python
summary_2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MLR</th>
      <th>GRNN</th>
      <th>PNN</th>
      <th>RBF</th>
      <th>ENSEMBLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.015443</td>
      <td>0.029372</td>
      <td>0.042082</td>
      <td>0.268967</td>
      <td>0.014786</td>
    </tr>
  </tbody>
</table>
</div>



So the lowest Mean Square Error (MSE) is again with Ensemble.
