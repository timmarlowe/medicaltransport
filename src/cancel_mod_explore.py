import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
import nltk.data
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from string import digits
import string

pd.options.mode.chained_assignment = None
plt.style.use('ggplot')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def time_split(X,y,var,year,month,day):
    split_date = pd.datetime(year,month,day)
    train_mask = X[var] <= split_date
    test_mask = X[var] > split_date
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    X_train.drop(var, axis=1,inplace=True)
    X_test.drop(var, axis=1, inplace=True)
    return X_train, X_test, y_train, y_test

def get_data(df):
    y = df['y']
    X = df[['Afternoon','Morning','CancelledHourOfPct','Evening','VehicleAmb',
            'CancelledDayOfPct','on_demand','EarlyMorning','CreatedDayOf',
            'estimatedCost','center_city_distance','to_from_distance',
            'CreatedHourOf','weekend','Completed_Count','rideNotes']]#,'pickup_day']]
    X['rideNotes'].fillna('',inplace=True)
    return X,y

def st_X(X_train, X_test):
    # standardize data
    standardizer = StandardScaler()
    standardizer.fit(X_train)
    X_train_std = standardizer.transform(X_train)
    X_test_std = standardizer.transform(X_test)
    return X_train_std, X_test_std

def smote(X, y, tp, k=None):
    """Generates new observations from the positive (minority) class.
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - [0, 1], target proportion of positive class observations

    Returns
    -------
    X_smoted : ndarray - 2D
    y_smoted : ndarray - 1D
    """
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
    and counts up the number in each.

    Parameters
    ----------
    X : ndarray - 2D
    y : ndarray - 1D

    Returns
    -------
    negative_count : Int
    positive_count : Int
    X_positives    : ndarray - 2D
    X_negatives    : ndarray - 2D
    y_positives    : ndarray - 1D
    y_negatives    : ndarray - 1D
    """
    negatives, positives = y == 0, y == 1
    negative_count, positive_count = np.sum(negatives), np.sum(positives)
    X_positives, y_positives = X[positives], y[positives]
    X_negatives, y_negatives = X[negatives], y[negatives]
    return negative_count, positive_count, X_positives, \
           X_negatives, y_positives, y_negatives

def k_folds_CV(model, X, y, n_splits=5):
    accuracy = []
    precision = []
    recall = []
    auc = []
    kf = KFold(n_splits,shuffle=True)
    for train_index, test_index in kf.split(X):
        Xk_train, Xk_test = X[train_index], X[test_index]
        yk_train, yk_test = y[train_index], y[test_index]
        model.fit(Xk_train, yk_train)
        preds = model.predict(Xk_test)
        # pdb.set_trace()
        accuracy.append(accuracy_score(yk_test, preds))
        precision.append(precision_score(yk_test,preds))
        recall.append(recall_score(yk_test,preds))
        auc.append(roc_auc_score(yk_test, preds))
    return np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(auc)

def run_and_print(model,X,y,title):
    cv_scores = k_folds_CV(model, X, y)
    print('\n{4}:\nAccuracy: {0:0.3f}\nPrecision: {1:0.3f}\nRecall: {2:0.3f}\nROC AUC: {3:0.3f}'.format(cv_scores[0],cv_scores[1],cv_scores[2],cv_scores[3],title))

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    no_punc = str.maketrans('','',string.punctuation)
    text_no_punc = text.translate(no_punc)
    tokens1 = nltk.word_tokenize(text_no_punc)
    remove_digits = str.maketrans('', '', digits)
    tokens = [t.translate(remove_digits) for t in tokens1]
    tokens = filter(None, tokens)
    no_stop = [t for t in tokens if t not in stop_words]
    stems = stem_tokens(no_stop, stemmer)
    return no_stop

def fit_nmf(r,data):
    nmf = NMF(n_components=r)
    nmf.fit(data)
    W = nmf.transform(data)
    H = nmf.components_
    return nmf.reconstruction_err_

def plot_nmf(error, num_topics):
    plt.plot(range(1,num_topics), error)
    plt.xticks(range(1, num_topics))
    plt.xlabel('number of latent topics')
    plt.ylabel('Reconstruction Error')
    plt.title('NMF Reconstruction Error by Number of Topics')
    plt.savefig('images/nmf_reconstruction_error.png')
    plt.close()

def topics(H, vocabulary):
    '''
    Print the most influential words of each latent topic, and prompt the user
    to label each topic. The user should use their humanness to figure out what
    each latent topic is capturing.
    '''
    hand_labels = []
    word_lsts = []
    for i, row in enumerate(H):
        top_five = np.argsort(row)[::-1][:5]
        print('topic', i)
        print('-->', ' '.join(vocabulary[top_five]))

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

def feature_importance_plot(importance, var_names, name, save):
    importance, var_names = zip(*sorted(zip(importance,var_names),reverse=True))
    fig, ax = plt.subplots(figsize=(14,8))
    x = np.arange(len(importance))
    plt.bar(x,importance)
    plt.xticks(x,var_names,rotation='vertical')
    plt.title('{}'.format(name))
    plt.tight_layout()
    plt.savefig('{}'.format(save))
    plt.close()

if __name__=='__main__':
    #Pre-processing
    ride_df = pd.read_pickle('data/model_df.pkl') #Load Dataframe
    X,y = get_data(ride_df)
    X_train, X_test, y_train, y_test = train_test_split(X,y)


    #ID Models I want to run
    model_list =    [LogisticRegression(),LogisticRegression(penalty='l1'),
                    RandomForestClassifier(n_jobs=-1),RandomForestClassifier(n_jobs=-1,n_estimators=1000,min_samples_leaf=2),
                    GradientBoostingClassifier(),GradientBoostingClassifier(n_estimators=100,min_samples_leaf=2,max_depth=5,learning_rate=.01),
                    AdaBoostClassifier(),AdaBoostClassifier(n_estimators=100,learning_rate=.01)]
    title_list =    ['Logistic Regression Base Model','Logistic Regression - L1 Penalty Added',
                    'Random Forest Classifier Base Model','Random Forest - Higher N-estimators, pruning trees',
                    'Gradient Boosting Classifier Base Model','Gradient Boosting Classifier - Higher n_estimators and min_samples changed',
                    'Adaptive Boosting Classifier Base Model','Adaptive Boosting Classifier - Higher n_estimators and learning rate changed']


    ##Start NLP modeling using ridenotes column

    vectorizer = TfidfVectorizer(tokenizer=tokenize, sublinear_tf=True)
    vectors_train = vectorizer.fit_transform(X_train['rideNotes'].values).toarray()
    vectors_test = vectorizer.transform(X_test['rideNotes'].values).toarray()
    vocabulary = vectorizer.get_feature_names()
    vocabulary = np.array(vocabulary)

    #NMF topic modeling of vectors
    # error = [fit_nmf(i,vectors_train) for i in range(1,26)]
    # plot_nmf(error,26)

    #Id'd 20 latent topics as as good as any - no real elbow - Use to create new feature set
    nmf = NMF(n_components=20, max_iter=100, random_state=12345, alpha=0.0)#Instantiating model
    W = nmf.fit_transform(vectors_train)#Fit-transforming train set
    H = nmf.components_#Id-ing latent topics
    topics(H, vocabulary) #hand_labeling topic values

    #Time to run new models
    run_and_print(MultinomialNB(),vectors_train,y_train,'Multinomial Naive Bayes Baseline')#Naive Bayes on this wordset - any better with just words?

    #Finalize both X_train and X_test for model set 2:
    mnb = MultinomialNB()
    mnb.fit(vectors_train,y_train)
    X_train['text_pred'] = mnb.predict_proba(vectors_train)[:,1]
    X_train.drop('rideNotes',axis=1, inplace=True)
    X_test['text_pred'] = mnb.predict_proba(vectors_test)[:,1]
    X_test.drop('rideNotes',axis=1, inplace=True)
    X_train_std, X_test_std = st_X(X_train, X_test) #standardizing
    X_res_train, y_res_train = smote(X_train_std, y_train,.5) #Resampling X and y (SMOTE)

    #Run Models from models list
    for i in range(len(model_list)):
         run_and_print(model_list[i], X_res_train, y_res_train, title_list[i])

    #Best cross-validated accuracy is in the Random forest run on the dataset with 20 MNF features. Use that as test set and model
    rf_final = RandomForestClassifier(n_jobs=-1,n_estimators=1000,min_samples_leaf=2)
    rf_final.fit(X_res_train, y_res_train)
    preds = rf_final.predict(X_test_std)
    print('Final Model - RF with 1000 trees:')
    print('Accuracy: {0:0.3f}'.format(accuracy_score(y_test, preds)))
    print('Precision: {0:0.3f}'.format(precision_score(y_test, preds)))
    print('Recall: {0:0.3f}'.format(recall_score(y_test, preds)))
    auc = roc_auc_score(y_test, preds)
    print('Area Under Curve: {0:0.3f}'.format(auc))
    probs = rf_final.predict_proba(X_test_std)[:,1]
    roc_curve(probs, y_test, auc, 'Cancellation Predictions','images/roc_saferide_cancel.png')

    #Finding Feature Importances in model
    cols = list(X_train.columns)
    importance = rf_final.feature_importances_
    feature_importance_plot(importance, cols,
                            'Feature Importance for Random Forest Ride Cancellation Classifier',
                            'images/feature_importance2.png' )

    #Graphing Coefficient values in Logistic Regression
    lr_final = LogisticRegression()
    lr_final.fit(X_res_train, y_res_train)
    coeffs = list(lr_final.coef_.flat)
    feature_importance_plot(coeffs, cols, 'Coefficient Values for Logistic Regression Ride Cancellation Classifier','images/logit_coeffs2.png' )
