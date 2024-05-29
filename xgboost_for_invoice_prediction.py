import os, pickle, json
import numpy as np
import pandas as pd
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.preprocessing import LabelEncoder

def convert_to_usd(curr, amount):
    if(curr=='USD'):
        return amount
    else:
        if(curr=='CAD'):
            return amount*0.73

def get_class_type(x):
    try:
        days = int(x)

        class_type = ''
        if(days<0):
            class_type = 'Early'
        if(days==0):
            class_type='OnTime'
        if(days>0 and days<=30):
            class_type = 'Late'
        if(days>30 and days<=90):
            class_type = 'VeryLate'
        if(days>90):
            class_type = 'CriticallyLate'
        
        return class_type
    except Exception as e:
        return 'NoClass'

def get_sector(x):
    if(x in ['Food', 'Retail']):
        return x
    else:
        return 'AllSectors'
    
def get_no_of_pays_per_type(data_df, cust_number, issue_date_format, pay_type='early'):

    print("called no of pay")
    df = data_df[data_df['cust_number']==cust_number]
    count = 0
    for index, row in df.iterrows():
        if(row['issue_date_format']<issue_date_format and row['payment_category']==pay_type):
            count = count + 1
    return count


def get_avg_amount_of_pays_per_type(data_df, cust_number, issue_date_format, pay_type='early'):

    print("called avg of pay")

    df = data_df[data_df['cust_number']==cust_number]
    count = 0
    amount = 0
    for index, row in df.iterrows():
        if(row['issue_date_format']<issue_date_format and row['payment_category']==pay_type):
            count = count + 1
            amount = amount + row['total_open_amount']

    return amount/(count+0.1)



def load_and_feature_engineering_on_test():

    hist_dump_df = pd.read_csv('invoice_dataset.csv')
    df = pd.read_csv('invoice_dataset_test.csv')
    print(df.columns)

    df['issue_date_format'] = pd.to_datetime(df['issue_date'], format='mixed')
    df['due_in_date_format'] = pd.to_datetime(df['due_in_date'], format='mixed')
    print("data shape before ", df.shape)

    data_df = df
    #ground truth data
    data_df['sector_format'] = data_df['Industry_Sector'].apply(lambda x: get_sector(x))

    #we need to understand the no of days due for payment
    data_df['no_of_days_for_due_pay'] = (data_df['due_in_date_format'] - data_df['issue_date_format']).dt.days

    #converting to US Dollars so that we have a same scale
    data_df['US_Dollars'] = data_df.apply(lambda x: convert_to_usd('USD', x.total_open_amount), axis=1)

#    data_df = data_df[:1000]
    # We are obtaining this info because we need to differentiatte between these classes

    #we will obtain the no of Early payments
    data_df['no_of_early_pays'] = data_df.apply(lambda x: get_no_of_pays_per_type(hist_dump_df, x.cust_number, x.issue_date_format, pay_type='Early'),axis=1)

    #we will obtain avg amount of Early payments
    data_df['avg_of_early_pays'] = data_df.apply(lambda x: get_avg_amount_of_pays_per_type(hist_dump_df, x.cust_number, x.issue_date_format, pay_type='Early'),axis=1)


    
    #we will obtain the no of OnTime payments
    data_df['no_of_OnTime_pays'] = data_df.apply(lambda x: get_no_of_pays_per_type(hist_dump_df, x.cust_number, x.issue_date_format, pay_type='OnTime'),axis=1)

    #we will obtain avg amount of OnTime payments
    data_df['avg_of_OnTime_pays'] = data_df.apply(lambda x: get_avg_amount_of_pays_per_type(hist_dump_df, x.cust_number, x.issue_date_format, pay_type='OnTime'),axis=1)

    #we will obtain the no of late payments
    data_df['no_of_late_pays'] = data_df.apply(lambda x: get_no_of_pays_per_type(hist_dump_df, x.cust_number, x.issue_date_format, pay_type='Late'),axis=1)

    #we will obtain avg amount of late payments
    data_df['avg_of_late_pays'] = data_df.apply(lambda x: get_avg_amount_of_pays_per_type(hist_dump_df, x.cust_number, x.issue_date_format, pay_type='Late'),axis=1)

    #we will obtain the no of VeryLate payments
    data_df['no_of_VeryLate_pays'] = data_df.apply(lambda x: get_no_of_pays_per_type(hist_dump_df, x.cust_number, x.issue_date_format, pay_type='VeryLate'),axis=1)

    #we will obtain avg amount of VeryLate payments
    data_df['avg_of_VeryLate_pays'] = data_df.apply(lambda x: get_avg_amount_of_pays_per_type(hist_dump_df, x.cust_number, x.issue_date_format, pay_type='VeryLate'),axis=1)

    #we will obtain the no of CriticallyLate payments
    data_df['no_of_CriticallyLate_pays'] = data_df.apply(lambda x: get_no_of_pays_per_type(hist_dump_df, x.cust_number, x.issue_date_format, pay_type='CriticallyLate'),axis=1)

    #we will obtain avg amount of CriticallyLate payments
    data_df['avg_of_CriticallyLate_pays'] = data_df.apply(lambda x: get_avg_amount_of_pays_per_type(hist_dump_df, x.cust_number, x.issue_date_format, pay_type='CriticallyLate'),axis=1)

    data_df.to_csv('data_feature_engineered_for_test.csv')

    return data_df


def load_and_transform_on_test():

    data_df = load_and_feature_engineering_on_test()

    with open('sector_label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    data_df['sector_label'] = le.transform(data_df['sector_format'])

    final_df = data_df[['no_of_days_for_due_pay',	'US_Dollars',	'no_of_early_pays',
                    'avg_of_early_pays'	,'no_of_OnTime_pays',	'avg_of_OnTime_pays',\
                    'no_of_late_pays',	'avg_of_late_pays',	'no_of_VeryLate_pays',	\
                    'avg_of_VeryLate_pays',	'no_of_CriticallyLate_pays',	'avg_of_CriticallyLate_pays', \
                    'sector_label']]

    infer_df = final_df

    infer_df.to_csv("Inference.csv")
    X = infer_df.to_numpy()
    print(X.shape)

    return X, infer_df


def infer_the_xgboost_for_invoice_prediction():

    #pass the input array of your choice it should have 13 dimenstions
    with open('xgb_invoice.pkl', 'rb') as f:
        clf = pickle.load(f)

    input_array, input_df = load_and_transform_on_test()
    resp = clf.predict(input_array)
    input_df['Classification'] = resp

    result_list = input_df['Classification'].to_list()

    input_df.to_csv('Real_time_classification_result.csv')

    return result_list

def load_and_feature_engineering():
    df = pd.read_csv('invoice_dataset.csv')
    print(df.columns)

    df['issue_date_format'] = pd.to_datetime(df['issue_date'], format='mixed')
    df['due_in_date_format'] = pd.to_datetime(df['due_in_date'], format='mixed')
    df['clear_date_format'] = pd.to_datetime(df['clear_date'], format='mixed')
    df['time_taken_for_payment']= (df['clear_date_format'] - df['due_in_date_format']).dt.days 
    print("data shape before ", df.shape)
    data_df = df.dropna(subset=['time_taken_for_payment'])
    print("new data shape ", data_df.shape)

    #ground truth data
    data_df['payment_category'] = data_df['time_taken_for_payment'].apply(lambda x : get_class_type(x))

    data_df['sector_format'] = data_df['Industry_Sector'].apply(lambda x: get_sector(x))

    #we need to understand the no of days due for payment
    data_df['no_of_days_for_due_pay'] = (data_df['due_in_date_format'] - data_df['issue_date_format']).dt.days

    #converting to US Dollars so that we have a same scale
    data_df['US_Dollars'] = data_df.apply(lambda x: convert_to_usd(x.invoice_currency, x.total_open_amount), axis=1)


    data_df = data_df[:1000]
    # We are obtaining this info because we need to differentiatte between these classes

    #we will obtain the no of Early payments
    data_df['no_of_early_pays'] = data_df.apply(lambda x: get_no_of_pays_per_type(data_df, x.cust_number, x.issue_date_format, pay_type='Early'),axis=1)

    #we will obtain avg amount of Early payments
    data_df['avg_of_early_pays'] = data_df.apply(lambda x: get_avg_amount_of_pays_per_type(data_df, x.cust_number, x.issue_date_format, pay_type='Early'),axis=1)


    
    #we will obtain the no of OnTime payments
    data_df['no_of_OnTime_pays'] = data_df.apply(lambda x: get_no_of_pays_per_type(data_df, x.cust_number, x.issue_date_format, pay_type='OnTime'),axis=1)

    #we will obtain avg amount of OnTime payments
    data_df['avg_of_OnTime_pays'] = data_df.apply(lambda x: get_avg_amount_of_pays_per_type(data_df, x.cust_number, x.issue_date_format, pay_type='OnTime'),axis=1)

    #we will obtain the no of late payments
    data_df['no_of_late_pays'] = data_df.apply(lambda x: get_no_of_pays_per_type(data_df, x.cust_number, x.issue_date_format, pay_type='Late'),axis=1)

    #we will obtain avg amount of late payments
    data_df['avg_of_late_pays'] = data_df.apply(lambda x: get_avg_amount_of_pays_per_type(data_df, x.cust_number, x.issue_date_format, pay_type='Late'),axis=1)

    #we will obtain the no of VeryLate payments
    data_df['no_of_VeryLate_pays'] = data_df.apply(lambda x: get_no_of_pays_per_type(data_df, x.cust_number, x.issue_date_format, pay_type='VeryLate'),axis=1)

    #we will obtain avg amount of VeryLate payments
    data_df['avg_of_VeryLate_pays'] = data_df.apply(lambda x: get_avg_amount_of_pays_per_type(data_df, x.cust_number, x.issue_date_format, pay_type='VeryLate'),axis=1)

    #we will obtain the no of CriticallyLate payments
    data_df['no_of_CriticallyLate_pays'] = data_df.apply(lambda x: get_no_of_pays_per_type(data_df, x.cust_number, x.issue_date_format, pay_type='CriticallyLate'),axis=1)

    #we will obtain avg amount of CriticallyLate payments
    data_df['avg_of_CriticallyLate_pays'] = data_df.apply(lambda x: get_avg_amount_of_pays_per_type(data_df, x.cust_number, x.issue_date_format, pay_type='CriticallyLate'),axis=1)



    #we will obtain the total no of  payments made
    #data_df['no_of_total_pays'] = ''

    #we will obtain Avg amount of  payments made
    #data_df['avg_of_total_pays'] = ''

    #avg no of due days in a early payments

    #avg no of due days in a ontime payments

    #avg no of due days in a late payments

    #avg no of due days in a verylate payments

    #avg no of due days in a critically payments



    data_df.to_csv('data_formatted.csv')

    return data_df
    

def dummy_data_generator():
    x = 'invoice_id	cust_number	name_customer	Industry_Sector	buisness_year	issue_date	due_in_date	invoice_currency	total_open_amount'
    list_of_cols = x.split('\t')

    num_of_json = 10
    json_list = []

    choice_list = [i for i in range(10000)]
    for i in range(num_of_json):
        temp_dict = {}
        for val in list_of_cols:
            temp_dict[str(val)] = str(np.random.choice(choice_list))
        json_list.append(temp_dict)
    
    with open('input_json_for_inference.json', 'w') as f:
        json.dump(json_list, f)

def real_time_infer_using_json(file_path):
    x = 'invoice_id	cust_number	name_customer	Industry_Sector	buisness_year	issue_date	due_in_date	invoice_currency	total_open_amount'
    list_of_cols = x.split('\t')
    print(list_of_cols)

    if(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        test_list = []
        for json_obj in data:
            input_list = []
            for val in list_of_cols:
                input_list.append(json_obj[val])

            test_list.append(input_list)
        df  = pd.DataFrame(test_list, columns = list_of_cols)
        df.to_csv('invoice_dataset_test.csv')
        result_list = infer_the_xgboost_for_invoice_prediction()


        with open('sector_label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)

        result_list_label = le.inverse_transform(result_list)

        final_result = []
        for i in range(len(data)):
            temp_dict = data[i]
            temp_dict['invoice-class'] = result_list_label[i]
            final_result.append(temp_dict)

        with open('result.json', 'w') as f:
            json.dump(final_result, f)
            

def load_and_transform():

    data_df = load_and_feature_engineering()

    le = LabelEncoder()

    le.fit(data_df['sector_format'])

    with open('sector_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    data_df['sector_label'] = le.transform(data_df['sector_format'])

    le2 = LabelEncoder()

    le2.fit(data_df['payment_category'])

    data_df['ground_truth_labels'] = le2.transform(data_df['payment_category'])

    final_df = data_df[['no_of_days_for_due_pay',	'US_Dollars',	'no_of_early_pays',
                    'avg_of_early_pays'	,'no_of_OnTime_pays',	'avg_of_OnTime_pays',\
                    'no_of_late_pays',	'avg_of_late_pays',	'no_of_VeryLate_pays',	\
                    'avg_of_VeryLate_pays',	'no_of_CriticallyLate_pays',	'avg_of_CriticallyLate_pays', \
                    'ground_truth_labels', 'sector_label']]

    test_df = final_df['ground_truth_labels']
    train_df = final_df.drop('ground_truth_labels', axis=1)

    train_df.to_csv("Training_dataset.csv")
    test_df.to_csv("Test_dataset.csv")
    X = train_df.to_numpy()
    Y = test_df.to_numpy()

    print(X.shape)
    print(Y.shape)

    return X, Y


def train_and_test_xgboost_for_invoice_prediction():
    X, y = load_and_transform()
    limit = int((X.shape[0])*0.8)
    print("Train set size is ", limit)

    X_train, X_test = X[:limit], X[limit:]
    y_train, y_test = y[:limit], y[limit:]

    print(X_train.shape)
    print(y_train.shape)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0).fit(X_train, y_train)
   #print(y_test)
    print("Obtained score is ", clf.score(X_test, y_test))

    with open('xgb_invoice.pkl', 'wb') as f:
        pickle.dump(clf, f)

    return clf


def train_and_test_xgboost():
    X, y = make_hastie_10_2(random_state=0)
    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    print(X_train.shape)
    print(y_train.shape)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0).fit(X_train, y_train)
    print(y_test)
    print("Obtained score is ", clf.score(X_test, y_test))

    with open('xgb.pkl', 'wb') as f:
        pickle.dump(clf, f)

    return clf


def read_and_infer_play():

    while(1):
        n = int(input("Enter number of elements : "))
        
        lst = []
        # iterating till the range
        for i in range(0, n):
            ele = int(input())
            # adding the element
            lst.append(ele)  
        
        print(f"your Input array is {lst}")
        input_array = lst
        np_array = np.array(input_array)
        np_array = np_array.reshape(1,-1)
        infer_the_xgboost_for_invoice_prediction(np_array)


def infer_xgboost(clf, input_array):

    print(clf.predict(input_array))
    return

def infer_the_xgboost_for_invoice_prediction_sample(input_array):

    #pass the input array of your choice it should have 13 dimenstions
    with open('xgb_invoice.pkl', 'rb') as f:
        clf = pickle.load(f)

    resp = clf.predict(input_array)
    print("class type ", resp)
  


if(__name__=="__main__"):
    '''
    clf = train_and_test_xgboost()
    test_input = np.array([0 for i in range(10)])
    test_input = test_input.reshape(1,-1)
    print(test_input.shape)
    infer_xgboost(clf, test_input)
    '''
    #load_and_feature_engineering()
    #load_and_transform()
    #train_and_test_xgboost_for_invoice_prediction()
    '''
    input_array = [i for i in range(13)]
    np_array = np.array(input_array)
    np_array = np_array.reshape(1,-1)
    infer_the_xgboost_for_invoice_prediction(np_array)
    '''
    #train_and_test_xgboost_for_invoice_prediction()
    dummy_data_generator()
    real_time_infer_using_json(file_path='input_json_for_inference.json')