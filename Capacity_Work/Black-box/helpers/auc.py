from sklearn import metrics
from sklearn.metrics import auc 
import numpy as np

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  
# Noise-cifar10
scores = np.array([-5.805, -3.492, -4.404, -4.282, -5.192, -6.729, -6.071, -3.918, -3.245, -4.831, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])  
# Exponential-weighting-mnist
scores = np.array([-2.526, -2.404, -2.355, -2.491, -2.473, -2.526, -2.471, -2.338, -2.434, -2.608, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
# Unrelated-mnist
scores = np.array([-3.070, -2.768, -2.466, -2.434, -2.705, -2.523, -2.584, -2.571, -2.794, -2.548, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]) 
fpr, tpr, thresholds = metrics.roc_curve(y, scores)
auc = metrics.auc(fpr, tpr)
print(fpr, tpr)
print(auc)