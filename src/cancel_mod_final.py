import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle

pd.options.mode.chained_assignment = None
plt.style.use('ggplot')

def get_data(df):
    y = df['y']
    X = df[['Afternoon','Morning','CancelledHourOfPct','Evening','VehicleAmb',
            'CancelledDayOfPct','on_demand','EarlyMorning','CreatedDayOf',
            'estimatedCost','center_city_distance','to_from_distance',
            'CreatedHourOf','weekend','Completed_Count','rideNotes']]#,'pickup_day']]
    X['rideNotes'].fillna('',inplace=True)
    return X,y

def smote(X, y, tp, k=None):
    """Generates new observations from the positive (minority) class.
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf"""

    if tp < np.mean(y):
        return X, y
    if k is None:
        k = int(len(X) ** 0.5)

    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_pos, y_pos)
    neighbors = knn.kneighbors(return_distance=False)

    positive_size = (tp * neg_count) / (1 - tp)
    smote_num = int(positive_size - pos_count)

    rand_idxs = np.random.randint(0, pos_count, size=smote_num)
    rand_nghb_idxs = np.random.randint(0, k, size=smote_num)
    rand_pcts = np.random.random((smote_num, X.shape[1]))
    smotes = []
    for r_idx, r_nghb_idx, r_pct in zip(rand_idxs, rand_nghb_idxs, rand_pcts):
        rand_pos, rand_pos_neighbors = X_pos[r_idx], neighbors[r_idx]
        rand_pos_neighbor = X_pos[rand_pos_neighbors[r_nghb_idx]]
        rand_dir = rand_pos_neighbor - rand_pos
        rand_change = rand_dir * r_pct
        smoted_point = rand_pos + rand_change
        smotes.append(smoted_point)

    print('Original classification counts of y_train {}'.format(Counter(y)))
    X_smoted = np.vstack((X, np.array(smotes)))
    y_smoted = np.concatenate((y, np.ones((smote_num,))))
    print('New classification counts of resampled y_train using SMOTE {}'.format(Counter(y_smoted)))
    return X_smoted, y_smoted

def div_count_pos_neg(X, y):
    """Helper function to divide X & y into positive and negative classes
    and counts up the number in each."""

    negatives, positives = y == 0, y == 1
    negative_count, positive_count = np.sum(negatives), np.sum(positives)
    X_positives, y_positives = X[positives], y[positives]
    X_negatives, y_negatives = X[negatives], y[negatives]
    return negative_count, positive_count, X_positives, \
           X_negatives, y_positives, y_negatives

def roc_curve(probabilities, labels, auc, name, save):
    TPR = []
    FPR = []
    thresholds = np.sort(probabilities)
    for prob in thresholds:
        predictions = (probabilities > prob).astype(int)
        tp = ((predictions == 1) & (labels == 1)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        tpr = tp/(fn + tp)
        fpr = fp/(tn + fp)
        FPR.append(fpr)
        TPR.append(tpr)
    fig, ax = plt.subplots()
    x = np.arange(1,101)/100
    y = np.arange(1,101)/100
    ax.plot(x,y)
    ax.plot(FPR, TPR,label='ROC Curve - Area Under Curve: {0:0.3f}'.format(auc))
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of {}".format(name))
    plt.legend()
    plt.savefig("{}".format(save))
    plt.close()

class CancelModel(object):

    def __init__(self):
        self._classifier = RandomForestClassifier(n_jobs=-1,n_estimators=1000,min_samples_leaf=2)
        self._standardizer = StandardScaler(with_mean=True)
        self._vectorizer = TfidfVectorizer(stop_words='english')
        self._bayespredictor = MultinomialNB()

    def smoter(self,X,y):
        return smote(X,y,.5)

    def fit(self,X,y):
        vectors = self._vectorizer.fit_transform(X.pop('rideNotes'))
        self._bayespredictor.fit(vectors,y)
        X['text_pred'] = self._bayespredictor.predict_proba(vectors)[:,1]
        X_scaled = self._standardizer.fit_transform(X)
        X_smoted, y_smoted = self.smoter(X_scaled,y)
        self._classifier.fit(X_smoted, y_smoted)
        return self

    def predict(self,X):
        vectors = self._vectorizer.transform(X.pop('rideNotes'))
        X['text_pred'] = self._bayespredictor.predict_proba(vectors)[:,1]
        X_scaled = self._standardizer.transform(X)
        preds = self._classifier.predict(X_scaled)
        return preds

    def predict_proba(self,X):
        vectors = self._vectorizer.transform(X.pop('rideNotes'))
        X['text_pred'] = self._bayespredictor.predict_proba(vectors)[:,1]
        X_scaled = self._standardizer.transform(X)
        preds = self._classifier.predict_proba(X_scaled)[:,1]
        return preds

    def scorer(self,X,y):
        preds = self.predict(X)
        recall = recall_score(y,preds)
        precision = precision_score(y,preds)
        auc = roc_auc_score(y,preds)
        accuracy = accuracy_score(y,preds)
        return recall, precision, auc, accuracy

if __name__=='__main__':
    #Preprocessing Data
    ride_df = pd.read_pickle('data/model_df.pkl') #Load Dataframe
    X,y = get_data(ride_df)
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    #Running Cancel model
    cm = CancelModel()
    cm.fit(X_train.copy(), y_train)
    preds = cm.predict(X_test.copy())
    probs = cm.predict_proba(X_test.copy())
    precision, recall, auc, accuracy = cm.scorer(X_test.copy(), y_test)
    roc_curve(probs, y_test, auc, 'Cancellation Predictions','images/roc_cancel_day_of.png')

#Pickling fit model
with open('data/cancel_model.pkl', 'wb') as f:
    pickle.dump(cm, f)
