import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sklearn.pipeline
from sklearn.preprocessing import Imputer

#define a function to make scatter plot
def scatter_plot(data, xname, yname, title, color = "blue"):
    ## input:
    # data: original dataset
    # xname: variable name used as x axis
    # yname: variable name used as y aixs
    # title: title of the plot
    ## output: scatterplot saved in local storage. 
    plt.figure()
    x = data[xname]
    y = data[yname]
    plt.scatter(x,y,c = color)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title.split(".")[0])
    plt.savefig(title, bbox_inches='tight')

#define a function to create new features
def create_feature(data, lab1, lab2, new_lab):
    ## input:
    # data: original dataset
    # lab1: variable name in the original dataset that would be used
    # lab2: variable name in the original dataset that would be used
    # new_lab: new variable name
    ## output: new dataset with the new variable
    for names in data.keys():
        if data[names][lab1] == 'NaN' or data[names][lab2] == 'NaN':
            data[names][new_lab] = 0
        else:
            data[names][new_lab] = float(data[names][lab1])/float(data[names][lab2])
    return data

#change data format, shifting NaN into 0.
#extract poi labels and features
def change_format(data, features):
    ##input: 
    # data: original data
    # features: features that you wish to choose
    ##output: 
    # dataset 1: pure data without poi
    # dataset 2: poi label
    ## its function is to split into poi label and other data with selected features.
    final_list = []
    label_list = []
    for names in data.keys():
        temp_list = []
        for feature in features:
            try:
                data[names][feature]
            except KeyError:
                print "error:",names,feature,"not present"
                return
            value = data[names][feature]
            if feature == "poi":      # seperate poi label 
                label_list.append(float(value))
                continue
            elif value < 0:
                value = -value     #change negative values into positive ones.
            temp_list.append(float(value))
        final_list.append(np.array(temp_list))
    return np.array(final_list), np.array(label_list)

#define a function to scale value. In this function, NaN will remained NaN
#other numeric value will distributed between 0 and 1 according to minmaxscaler. 
def scaler(data, feature_num):
    #input: original data(list)
    #output: new data(same format)
    f = pd.DataFrame(features_data)
    i = 0
    j = 0
    list2 = []
    while j < len(features_data):
        h_1 = f.iloc[j]
        list1 = []
        i = 0
        while i <= feature_num-1:
            f_1 = f[i]
            j_1 = h_1[i]
            min_value = min(f_1[f_1.isnull() == False])
            max_value = max(f_1[f_1.isnull() == False])
            if j_1 == "NaN":
                list1.append("NaN")
            else:
                new_val = (j_1-min_value)/(max_value-min_value)
                list1.append(new_val)
            i = i+1
        list2.append(list1)
        j = j+1
    
    return list2

#select top 10 features having the closest relationship with poi label
def select_best_features(features_data, labels, features_list, number = 10):
    f1 = SelectKBest(chi2, k = number)
    f2 = f1.fit_transform(features_data, labels)
    scores = f1.scores_
    dic = {}
    features_list.remove("poi")
    for feature, score in zip(features_list, scores):
        dic[feature] = score
    new_dic = sorted(dic.iteritems(), key=lambda d:d[1], reverse = True)
    return new_dic[0:12]

#define a function to train and test, returning performance report
def report(clf, features_train, features_test, labels_train, labels_test):
    ##input: 
    # clf: classifier you set
    ##output: accuracy, recall, precision and f1 score you have got.
    steps = [('classifier', clf)]

    pipeline = sklearn.pipeline.Pipeline(steps)

    pipeline.fit(features_train, labels_train)

    y_prediction = pipeline.predict( features_test )

    report = sklearn.metrics.classification_report( labels_test, y_prediction )

    return report


#read dataset from local
with open("final_project_dataset.pkl", "r") as data_file:
    rawdata = pickle.load(data_file)
mydata = rawdata

#transfer data from dictionary to data frame.
#mydata_df is the dataframe transformed from original dataset
mydata_df = pd.DataFrame.from_dict(data = mydata, orient = 'index')
#exploration
print("Show the list of column names:")
print(list(mydata_df.columns.values))
print("Total number of data points:")
print(len(mydata))
print("Number of POIs:")
print(len(mydata_df[mydata_df.poi == True]))
print("Number of non-POIs:")
print(len(mydata_df[mydata_df.poi == False]))
#plot relationship between total stock value and total payments
scatter_plot(mydata_df, "salary",
             "bonus","salary_vs_bonus.png")

#drop TOTAL, the outlier, and scatter again
#mydata_df1 is the dataframe transformed from dataset without "TOTAL"
mydata.pop('TOTAL')
mydata_df1 = pd.DataFrame.from_dict(data = mydata, orient = 'index')
#mydata_df1 = mydata_df.drop("TOTAL")
scatter_plot(mydata_df1, "salary",
             "bonus","salary_vs_bonus2.png", color = mydata_df1['poi'])
print("scatter plot: salary vs bonus, has been done")

#add new features, fraction of sending to poi and fraction of receiving from poi
mydata = create_feature(mydata, "from_this_person_to_poi",\
                        "from_messages","fraction_to_poi")
mydata = create_feature(mydata, "from_poi_to_this_person",\
                        "to_messages", "fraction_from_poi")
print("features have been added")
#mydata is now a dataset without "TOTAL" row and having two more columns

#plot scatter plot: fraction to poi vs fraction from poi
#mydata_df2 is only used to plot this graph
mydata_df2 = pd.DataFrame.from_dict(data = mydata, orient = 'index')
scatter_plot(mydata_df2, "fraction_to_poi",
             "fraction_from_poi","fraction_to_poi_vs_fraction_from_poi.png", color = mydata_df2['poi'])
print("scatter plot: fraction_to_poi_vs_fraction_from_poi, has been done")

#count NaN number
#mydata_df3 is only used to count NaN number.
mydata_df3 = mydata_df2.drop(["email_address",'poi','fraction_from_poi','fraction_to_poi'],axis = 1)
missing_number = {}
for column in mydata_df3.columns.values:
    v = len(mydata_df3[mydata_df3[column] == 'NaN'])
    missing_number[column] = v
missing_number_df = pd.DataFrame.from_dict(data = missing_number, orient = 'index')
print("show the number of missing values:")
print(missing_number_df)

#set up a feature list.
#this is the simplified features. How they were selected are explained in pdf report.
features_list_sim2 = ["poi", 'salary','deferral_payments','deferred_income','director_fees',
                 'exercised_stock_options','expenses',
                 'fraction_from_poi','fraction_to_poi',
                 'loan_advances','long_term_incentive','other',
                 'restricted_stock','restricted_stock_deferred',
                 'shared_receipt_with_poi','total_payments',
                 'total_stock_value']

#get pure data and poi labels from mydata
features_data_sim, labels_data_sim = change_format(mydata, features_list_sim2)

#-----------Rank Features: DecisionTree method---------------------------------
#replacing NaN with median
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
features_data_scl1 = imp.fit_transform(features_data_sim)

#split data into train and test set for non-scaled data.
features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features_data_scl1, labels_data_sim, test_size=0.3, random_state = 170,stratify = labels_data_sim )

#select best features: decisiontree
clf = tree.DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(features_train, labels_train)
print("show feature scores by decision tree")
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > .00005:
        print "{}:{}".format(features_list_sim2[i+1],clf.feature_importances_[i] )
        
#------------Rank Features: SelectKBest method--------------------------------- 
#split data into train and test set for scaled data.
features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features_data_sim, labels_data_sim, test_size=0.3, random_state = 170,stratify = labels_data_sim )
#then scaled training data
features_train_scl = scaler(features_train, len(features_list_sim2)-1)
#then fill NaNs with median
features_data_scl2 = imp.fit_transform(features_data_scl)
#select best features: selectkbest 
best_data= select_best_features(features_data_scl2, labels_data_sim, features_list_sim2)
print("show features by selectkbest score:")
print(list(best_data))

#--------------Starts to train--------------------------------------------------
#refresh the 4 dataset with non-scaled data.
features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features_data_scl1, labels_data_sim, test_size=0.3, random_state = 170,stratify = labels_data_sim )

#replacing NaN with median in non-scaled data
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
features_data_scl2 = imp.fit_transform(features_data_sim)

#features I selected
features = ['poi','fraction_to_poi','exercised_stock_options','deferred_income','shared_receipt_with_poi']

clf2 = GaussianNB()

#print out report
print("Gaussian Naive Base Report")
print(report(clf2,features_train, features_test, labels_train, labels_test))

### dump your classifier, dataset and features_list so
### anyone can run/check your results
clf = clf2.fit(features_train, labels_train)
data_dict = mydata
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features, open("my_feature_list.pkl", "w") )
