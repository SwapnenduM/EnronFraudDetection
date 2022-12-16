# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("../tools/")

from time import time

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier

from sklearn import tree
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score,precision_score,accuracy_score


# In[2]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'deferral_payments',
                 'total_payments', 
                 'loan_advances', 
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income', 
                 'total_stock_value', 
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive', 
                 'restricted_stock',
                 'director_fees',
                 'to_messages',
                 'from_poi_to_this_person', 
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[3]:


print ("Scatter plot before removing outliers")   
for point in data_dict:
    salary = data_dict[point]["salary"]
    bonus = data_dict[point]["bonus"]
    plt.scatter( salary, bonus)


plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

### Task 2: Remove outliers
out=['TOTAL','LOCKHART EUGENE E','THE TRAVEL AGENCY IN THE PARK']

for i in out:
    data_dict.pop(i)
    
print ("Scatter plot after removing outliers")
for point in data_dict:
    salary = data_dict[point]["salary"]
    bonus = data_dict[point]["bonus"]
    plt.scatter( salary, bonus)


plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


# In[4]:


for point in data_dict:
    salary = data_dict[point]["from_poi_to_this_person"]
    bonus = data_dict[point]["from_this_person_to_poi"]
    plt.scatter( salary, bonus)


plt.xlabel("from_poi_to_this_person")
plt.ylabel("from_this_person_to_poi")
plt.show()


# In[5]:



from sklearn.feature_selection import SelectKBest,f_classif
f_scores =[]



def pf(data_dict):
    for k, v in data_dict.items():
#Assigning value to the feature 'proportion_from_poi'

        if v['from_poi_to_this_person'] != 'NaN' and  v['from_messages'] != 'NaN':
            v['proportion_from_poi'] = float(v['from_poi_to_this_person']) / v['from_messages'] 
        else:    
            v['proportion_from_poi'] = 0.0
    return (data_dict)       
            
def pt(data_dict):
    for k, v in data_dict.items():
        #Assigning value to the feature 'proportion_to_poi'        
        if v['from_this_person_to_poi'] != 'NaN' and  v['to_messages'] != 'NaN':
            v['proportion_to_poi'] = float(v['from_this_person_to_poi'] )/ v['to_messages']   
        else:
            v['proportion_to_poi'] = 0.0
    return (data_dict)


# In[6]:


def net_worth (data_dict) :
    features = ['total_payments','total_stock_value']
    
    for key in data_dict :
        name = data_dict[key]
        
        is_null = False 
        
        for feature in features:
            if name[feature] == 'NaN':
                is_null = True
        
        if not is_null:
            name['net_worth'] = name[features[0]] + name[features[1]]
        
        else:
            name['net_worth'] = 'NaN'
            
    return data_dict                
            
def select_features(features,labels,features_list,k=10) :
    clf = SelectKBest(f_classif,k)
    new_features = clf.fit_transform(features,labels)
    features_l=[features_list[i+1] for i in clf.get_support(indices=True)]
    f_scores = list(zip(features_list[1:],clf.scores_[:]))
    f_scores = sorted(f_scores,key=lambda x: x[1],reverse=True)
    return new_features, ['poi'] + features_l, f_scores


# In[7]:


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
data_dict = net_worth(data_dict)

data_dict=pf(data_dict)
data_dict=pt(data_dict)
features_list+=['net_worth','proportion_from_poi','proportion_to_poi']

my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

for point in data_dict:
    salary = data_dict[point]["proportion_from_poi"]
    bonus = data_dict[point]["proportion_to_poi"]
    plt.scatter( salary, bonus)

plt.xlabel("proportion_from_poi")
plt.ylabel("proportion_to_poi")
plt.show()


# In[8]:


features,features_list,f_scores=select_features(features,labels,features_list,k=6)
# call the function with uses selectkbest
print(("features_list---" ,features_list))
print("feature scores")
for i in f_scores:
    print (i)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[9]:


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf,my_dataset,features_list)


# In[10]:


from sklearn.tree import DecisionTreeClassifier
# clf1=tree.DecisionTreeClassifier()
# test_classifier(clf1,my_dataset,features_list)


# In[11]:


from sklearn.ensemble import AdaBoostClassifier
# clf2 = AdaBoostClassifier()
# test_classifier(clf2,my_dataset,features_list)


# In[12]:


# from sklearn.neighbors import KNeighborsClassifier
# clf3=KNeighborsClassifier(n_neighbors = 4)
# test_classifier(clf3,my_dataset,features_list)


# In[13]:


# from sklearn.neighbors.nearest_centroid import NearestCentroid
# clf4 = NearestCentroid()
# test_classifier(clf4,my_dataset,features_list)


# In[14]:


# from sklearn.ensemble import RandomForestClassifier
# clf5 = RandomForestClassifier()
# test_classifier(clf5,my_dataset,features_list)


# In[15]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
 
t = time()
pipe1 = Pipeline([('pca',PCA()),('classifier',GaussianNB())])
param = {'pca__n_components':[4,5,6]}
gsv = GridSearchCV(pipe1, param_grid=param,n_jobs=2,scoring = 'f1',cv=2)
gsv.fit(features_train,labels_train)
clf = gsv.best_estimator_
print(("GausianNB with PCA fitting time: %rs" % round(time()-t, 3)))
pred = clf.predict(features_test)

t = time()
test_classifier(clf,my_dataset,features_list,folds = 1000)
print(("GausianNB  evaluation time: %rs" % round(time()-t, 3)))


# In[16]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[17]:


#adbc = AdaBoostClassifier(random_state=40)
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)
#dt = []
#for i in range(6):
#    dt.append(DecisionTreeClassifier(max_depth=(i+1)))
#adbpara = {'base_estimator': dt,'n_estimators': [60,45, 101,10]}
#t = time()
#adbt = GridSearchCV(adbc, adbpara, scoring='f1',)
#adbt = adbt.fit(features_train,labels_train)
#print("AdaBoost fitting time: %rs" % round(time()-t, 3))
#adbc = adbt.best_estimator_
#t = time()
#test_classifier(adbc, data_dict, features_list, folds = 100)
#print("AdaBoost evaluation time: %rs" % round(time()-t, 3))


