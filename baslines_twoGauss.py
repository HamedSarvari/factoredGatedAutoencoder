# This script runs LOF baseline to compare the results on the two gaussinas toy dataset
#########################################
from universal_utils import *
from pyod.models import lof
from Autoencoder_utils_torch import eval_model
from sklearn.metrics import average_precision_score
#########################################
data_name = 'TwoGauss_data_7dim'
loaded_data = load_obj(data_name)
data = loaded_data['data']
labels = loaded_data['labels']
class_labels = loaded_data['class_labels']
##########################################

model= lof.LOF()
model.fit(data)
print(model.decision_scores_)

print(average_precision_score(labels,model.decision_scores_))

