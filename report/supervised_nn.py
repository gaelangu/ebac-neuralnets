
# coding: utf-8

# ## CA Assignment:
# #### Neural Network Ensembles
# ----
# 
# ##### Given :  Two benchmark classification/regression problems:
# -  Diabetes.csv 
# 
# The diabetes data set contains the diagnostic data to investigate whether the
# patient shows signs of diabetes according to World Health Organization criteria
# such as the 2-hour post-load plasma glucose.
# 
# -  Winequality-white.csv
# 
# 
# The winequality-white data is related to the white variants of the Portuguese
# "Vinho Verde" wine. The goal is to model wine quality based on physicochemical
# tests.
# 
# ##### Expected :
# 1. Train a group of different types of NNs using different NN tools to solve the two problems given. 
# (Use 2 different tools to train 2-3 different types of NNs)
# 2. Work on the two data sets
#  You may partition each data set into two subsets: eg 75% as training data and
# 25% as test data
# 3. Train the NNs to achieve the highest possible classification accuracy or lowest
# possible MSE.
# 4. NN ensemble - combine the outputs of individual NNs for final output 
# (you may define certain calculation, such as rule(s) for the integration)
# Compare the NN performance between the NN ensemble and the individual NNs
# 

# In[1]:

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
get_ipython().magic('matplotlib inline')


#  ## 1. [ Diabities Problem ]

# For Each Attribute: (all numeric-valued)
#    1. Number of times pregnant
#    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#    3. Diastolic blood pressure (mm Hg)
#    4. Triceps skin fold thickness (mm)
#    5. 2-Hour serum insulin (mu U/ml)
#    6. Body mass index (weight in kg/(height in m)^2)
#    7. Diabetes pedigree function
#    8. Age (years)
#    9. Class variable (0 or 1)
#    
#    
# Class Distribution: (class value 1 is interpreted as "tested positive for
#    diabetes")
# 
#    Class Value  Number of instances
#    0            500
#    1            268

# In[2]:

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


# In[3]:

#total
print("total size of records")
np.size(df_diab)


# In[4]:

X1 = df_diab.iloc[:,:8]
y1 = df_diab['cls']
X1,y1 = shuffle(X1,y1)
#X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(preprocessing.minmax_scale(X1),preprocessing.minmax_scale(y1),train_size=0.70)


# In[5]:

print("train",X1_train.shape)
print("test",X1_test.shape)
print("train_y",y1_train.shape)


# In[6]:


print(y1_train[0])
X1_train[0]


# In[ ]:




# In[ ]:




# In[7]:

X1_test[0]


# In[8]:

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

    
    


#  ## 1a.  GRNN Network - [ Diabities Problem ]

# In[9]:

grnn_nw = algorithms.GRNN(std=0.1, verbose=True)
print(grnn_nw)


# In[10]:

grnn_nw.train(X1_train, y1_train)


# In[11]:

y1_predicted = grnn_nw.predict(X1_test).round()

y1_predicted[0]


# In[12]:

#accuracy
estimators.rmse(y1_predicted, y1_test)


# In[13]:

#confusion matrix
confusion_matrix(y1_test,y1_predicted)


# In[14]:

from sklearn.metrics import accuracy_score
grnn_acc_score = accuracy_score(y1_test, y1_predicted)
print("Grnn accuracy score ", grnn_acc_score)


#  ## 1b.  PNN Network - [ Diabities Problem ]
# 

# In[15]:

pnn_nw = algorithms.PNN(std=10, verbose=False)
print(pnn_nw)


# In[16]:

pnn_nw.train(X1_train, y1_train)


# In[17]:

y1_pnn_predicted = pnn_nw.predict(X1_test).round()
y1_pnn_predicted[0]


# In[18]:

#accuracy
estimators.rmse(y1_pnn_predicted, y1_test)


# In[19]:

#confusion matrix
confusion_matrix(y1_test,y1_pnn_predicted)


# In[20]:

pnn_acc_score = accuracy_score(y1_test, y1_pnn_predicted)
print("Pnn accuracy score ", pnn_acc_score)


#  ## 1c.  RBF - [ Diabities Problem ]
# 
# 

# In[21]:

rbf_nw = algorithms.RBFKMeans(n_clusters=2, verbose=False)


# In[22]:

rbf_nw.train(X1_train, epsilon=1e-5)


# In[23]:

y1_rbf_predicted = rbf_nw.predict(X1_test)


# In[24]:

confusion_matrix(y1_test,y1_rbf_predicted)


# In[25]:

rbf_acc_score = accuracy_score(y1_test, y1_rbf_predicted)
print("RBF accuracy score ", rbf_acc_score)


# ## 1d. Ensemble Learning - [ Diabities Problem ]

# In[26]:

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


# In[27]:

mv_clf = mv_clf.fit(X1_train, y1_train)


# In[28]:

X1_test.shape
#X1_train.shape
#mv_clf
#y_mv_clf_predicted = mv_clf.predict(X1_test)


# In[29]:

y_mv_clf_predicted = mv_clf.predict(X1_test)


# In[30]:

y_mv_clf_predicted[0]


# In[31]:

confusion_matrix(y1_test,y_mv_clf_predicted)


# In[32]:

mv_clf_acc_score = accuracy_score(y1_test, y_mv_clf_predicted)
print("Ensembles accuracy score ", mv_clf_acc_score)


#  ## 1e.  MLP - [ Diabities Problem ]
# 
# 

# In[33]:

from sklearn.neural_network import MLPClassifier
mlp_nw = MLPClassifier(solver='lbfgs', alpha=0.01,max_iter=2000,  hidden_layer_sizes=(5, 2), random_state=1, activation='relu')
#sgd


# In[34]:

mlp_model = mlp_nw.fit(X1_train, y1_train)
mlp_model


# In[35]:

y1_mlp_predicted = mlp_model.predict(X1_test)


# In[36]:

y1_mlp_predicted[1]


# In[37]:


confusion_matrix(y1_test,y1_mlp_predicted)


# In[38]:

mlp_acc_score = accuracy_score(y1_test, y1_mlp_predicted)
print("Pnn accuracy score ", mlp_acc_score)


# In[ ]:




#  ## 1f.  MLFF with tensorflow - [ Diabities Problem ]
# 
# 

# In[39]:

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


# In[40]:

# tf Graph input
x = tf.placeholder("float", [None, n_input],name="x")
y = tf.placeholder("float", [None,n_classes],name="y")


# In[41]:

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


# In[42]:

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


# In[43]:

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


# In[44]:

# Construct model
pred = multilayer_perceptron_tf(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_1).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


# In[45]:

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
            print("Epoch:", '%04d' % (epoch+1), "cost=",                 "{:.9f}".format(c))
            errors.append(c)
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(tf.round(pred), 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    y1_test_h = one_hot(y1_test)
    
    mlp_tf_acc_Score = accuracy.eval({x: X1_test, y: y1_test_h})
    print("Accuracy:", mlp_tf_acc_Score)


# In[46]:

summary_1 = pd.DataFrame([[grnn_acc_score,pnn_acc_score,rbf_acc_score,mv_clf_acc_score,mlp_acc_score,mlp_tf_acc_Score]])
summary_1.columns=['GRNN', 'PNN', 'RBF', 'ENSEMBLE', 'MLP','MLP_TF']


# In[47]:

summary_1


# As seen above, the best classifier was with **Ensemble** learning.

# ## 2. Wine Quality

#  Input variables (based on physicochemical tests):
#    1. - fixed acidity
#    2. - volatile acidity
#    3. - citric acid
#    4. - residual sugar
#    5. - chlorides
#    6. - free sulfur dioxide
#    7. - total sulfur dioxide
#    8. - density
#    9. - pH
#    10. - sulphates
#    11. - alcohol
#    Output variable (based on sensory data): 
#    12. - quality (score between 0 and 10)

# In[48]:

df_wines = pd.read_csv('winequality-white.csv')


# In[49]:

df_wines.head(5)


# In[50]:

X2 = df_wines.iloc[:,:11]
y2 = df_wines['quality']
X2,y2 = shuffle(X2,y2)
#X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(preprocessing.minmax_scale(X2),preprocessing.minmax_scale(y2),train_size=0.70)


# In[51]:

print("train",X2_train.shape)
print("test",X2_test.shape)
print("train_y",y2_train.shape)


# ### 2a. MLR ( Multi linear Regression)

# In[52]:

from sklearn.neural_network import MLPRegressor
mlr_nw = MLPRegressor(solver='lbfgs', alpha=0.01,max_iter=2000,  hidden_layer_sizes=(5, 2), random_state=1, activation='relu')
#sgd


# In[53]:

mlr_model = mlr_nw.fit(X2_train, y2_train)
mlr_model


# In[54]:

y2_mlr_predicted = mlr_model.predict(X2_test)


# In[55]:

y2_mlr_predicted[0]


# In[56]:

# The mean squared error
y2_mlr_mse = np.mean((y2_mlr_predicted - y2_test) ** 2)
print("Mean squared error: %.4f"
      % y2_mlr_mse)


# In[57]:

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % mlr_model.score(X2_test, y2_test))


# In[58]:

# Plot outputs
#plt.scatter(X2_test[:,1:2], y2_test,  color='black')
#plt.plot(X2_test, y2_mlr_predicted, color='blue',
#         linewidth=3)

#plt.xticks(())
#plt.yticks(())


# ### 2b. GRNN

# In[ ]:




# In[59]:

grnn_nw_2 = algorithms.GRNN(std=0.1, verbose=True)
print(grnn_nw_2)


# In[60]:

grnn_nw_2.train(X2_train, y2_train)


# In[61]:

y2_grnn_predicted = grnn_nw_2.predict(X2_test)


# In[62]:

y2_grnn_predicted[0]


# In[63]:

# The mean squared error
y2_grnn_mse = np.mean((y2_grnn_predicted - y2_test) ** 2)
print("Mean squared error: %.4f"
      % y2_grnn_mse)


# ### 2c. PNN Network 

# In[64]:

pnn_nw_2 = algorithms.PNN(std=10, verbose=False)
print(pnn_nw_2)


# In[65]:

pnn_nw_2.train(X2_train, y2_train)


# In[66]:

y2_pnn_predicted = pnn_nw_2.predict(X2_test)


# In[67]:

y2_pnn_predicted[0]


# In[68]:

# The mean squared error
y2_pnn_mse = np.mean((y2_pnn_predicted - y2_test) ** 2)
print("Mean squared error: %.4f"
      % y2_pnn_mse)


# ### 2d. RBF Network

# In[69]:

rbf_nw_2 = algorithms.RBFKMeans(n_clusters=2, verbose=False)


# In[70]:

rbf_nw_2.train(X2_train, epsilon=1e-5)


# In[71]:

y2_rbf_predicted = rbf_nw_2.predict(X2_test)


# In[72]:

y2_rbf_predicted[0]


# In[73]:

#The mean squared error
y2_rbf_mse = np.mean((y2_rbf_predicted - y2_test) ** 2)
print("Mean squared error: %.4f"
     % y2_rbf_mse)


# :( Too high

# ### 2d. Ensembles

# In[74]:

from sklearn.ensemble import AdaBoostRegressor
#clf2 = RandomForestClassifier(random_state=1)
#clf3 = GaussianNB()
#clf4 = SVC(kernel='rbf', probability=True)

en_reg = AdaBoostRegressor(base_estimator=mlr_nw ,n_estimators=50)
#


# In[75]:

en_reg.fit(X2_train, y2_train)


# In[76]:

y2_ens_predicted = en_reg.predict(X2_test)


# In[77]:

y2_ens_predicted[0]


# In[78]:

# The mean squared error
y2_ens_mse = np.mean((y2_ens_predicted - y2_test) ** 2)
print("Mean squared error: %.4f"
      % y2_ens_mse)


# ### Summary

# In[79]:

summary_2 = pd.DataFrame([[y2_mlr_mse,y2_grnn_mse,y2_pnn_mse,y2_rbf_mse,y2_ens_mse]])
summary_2.columns=['MLR', 'GRNN', 'PNN', 'RBF','ENSEMBLE']


# In[80]:

summary_2


# So the lowest Mean Square Error (MSE) is again with Ensemble.
