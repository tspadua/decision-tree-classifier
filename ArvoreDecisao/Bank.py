import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff

def bank():
    data,meta = arff.loadarff('./bank.arff')

    attributes = meta.names()
    data_value = np.asarray(data)


    age = np.asarray(data['age']).reshape(-1,1)
    average = np.asarray(data['average']).reshape(-1,1)
    day = np.asarray(data['day']).reshape(-1,1)
    duration = np.asarray(data['duration']).reshape(-1,1)
    campaign = np.asarray(data['campaign']).reshape(-1,1)
    pdays = np.asarray(data['pdays']).reshape(-1,1)

    features = np.concatenate((age, average, day, duration, campaign, pdays),axis=1)
    target = data['subscribed']


    Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

    plt.figure(figsize=(10, 6.5))
    tree.plot_tree(Arvore,feature_names=['age', 'average','day', 'duration', 'campaign', 'pdays'],class_names=['yes', 'no'],
                   filled=True, rounded=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(25, 10))
    metrics.plot_confusion_matrix(Arvore,features,target,display_labels=['yes', 'no'], values_format='d', ax=ax)
    plt.show()

bank()