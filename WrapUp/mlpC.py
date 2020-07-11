from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.datasets import fetch_mldata
from mnist import load_mnist
import numpy as np
import pickle
import gzip
pca = PCA(n_components = 100)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 设置神经网络模型参数
train_data=x_train.reshape(60000,784)
test_data=x_test.reshape(10000,784)
pca.fit(train_data)
train_data_pca = pca.transform(train_data)
test_data_pca = pca.transform(test_data)
mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, n_iter_no_change = 100, tol = 1e-9, hidden_layer_sizes=(50,50), random_state=1, max_iter=5, verbose= True, learning_rate_init=.1)
mlp.fit(x_train,t_train)
pred = mlp.predict(x_test)
t_true = []
for i in range(t_test.shape[0]):
    for j in range(t_test.shape[1]):
        if t_test[i,j] == 1:
           t_true.append(j)
pred_true = []
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if pred[i,j] == 1:
            pred_true.append(j)
            break
        if j == 9:
            pred_true.append(np.random.randint(0,10))
print(len(pred_true))
print(pred.shape)
print(t_test.shape)
print(pred.shape)
print ("Classification Report")
print(classification_report(t_test, pred))
print ("Confusion Report")
print(confusion_matrix(t_true, pred_true))
print (mlp.score(x_test,t_test))
print (mlp.n_layers_)
print (mlp.n_iter_)
print (mlp.loss_)
print (mlp.out_activation_)

