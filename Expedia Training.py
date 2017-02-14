
# coding: utf-8

# In[4]:

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score


data = pd.read_csv("/home/fabricc/Desktop/filtered_expedia_proportional.csv")
train, test = train_test_split(data, test_size = 0.3)
#train.drop('starrating_diff', axis=1, inplace=True)
#train.drop('usd_diff', axis=1, inplace=True)


# In[5]:

#print train.head(10)


# In[6]:

removed_columns = ['Unnamed: 0','click_or_book','time','year','click_bool','booking_bool','position']

predictors = [c for c in train.columns if c not in removed_columns]

#predictors = ['site_id','prop_id','prop_starrating','prop_review_score',
                #'prop_location_score1','prop_location_score2','prop_log_historical_price',
                 # 'position', 'srch_length_of_stay','srch_query_affinity_score',
                  #    'orig_destination_distance','starrating_diff','visitor_location_country_id',
                   #       'prop_brand_bool','month','month_day','promotion_flag', 'price_usd', 'srch_adults_count']



# In[7]:

rfc_click = RandomForestClassifier(n_estimators=100,n_jobs=-1)

labels = train["click_bool"].values
features = train[predictors].values

rfc_click.fit(features,labels);

print "Features sorted by their score:"

print sorted(zip(map(lambda x: round(x, 4), rfc_click.feature_importances_), predictors), 
             reverse=True)

sol_click = rfc_click.predict(test[predictors])
sol_prob_click = rfc_click.predict_proba(test[predictors])

#print rfc_click.score(sol_click,test["click_bool"])

target_names = ['0', 'pred']
print(classification_report(test["click_bool"], sol_click, target_names=target_names))


# In[ ]:

rfc_book = RandomForestClassifier(n_estimators=100, n_jobs=-1)

labels = train["booking_bool"].values
features = train[predictors].values

rfc_book.fit(features,labels);

print "Features sorted by their score:"

print sorted(zip(map(lambda x: round(x, 4), rfc_book.feature_importances_), predictors), 
             reverse=True)

sol_book = rfc_book.predict(test[predictors])
sol_prob_book = rfc_book.predict_proba(test[predictors])


#print rfc_book.score(sol_book,test["booking_bool"])


target_names = ['0', 'pred']
print(classification_report(test["booking_bool"], sol_book, target_names=target_names))


# In[ ]:

joint_pred = []

click_prob = 0
click_pred = 0
book_prob = 0
book_pred = 0


for click, book in zip(sol_prob_click , sol_prob_book):
    if(click[0] > click[1]): 
        click_prob = click[0]
        click_pred = 0
    else: 
        click_prop = click[1]
        click_pred = 1
        
    if(book[0] > book[1]): 
        book_prob = book[0]
        book_pred = 0
    else:
        book_prop = book[1]
        book_pred = 1
        
    if(book_prob > click_prob):
        if(book_pred): 
            joint_pred.append(5)
        else: 
            joint_pred.append(0)
    else:
        if(click_pred):
            joint_pred.append(1)
        else:
            joint_pred.append(0)
        


# In[ ]:

target_names = ['0', 'click','book']
print(classification_report(test["click_or_book"], joint_pred, target_names=target_names))


# In[ ]:

def click_or_book_random_forest_estimation(data, X_train, X_test, y_train, y_test, predictors):

    rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

    #labels = train["click_or_book"].values
    #features = train[predictors].values
    
    labels = data[y_train]
    features = data[X_train[predictors]].values

    rfc.fit(features,labels);

    print "Features sorted by their score:"

    print sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), predictors), 
             reverse=True)

    sol = rfc.predict(data[X_test[predictors]])

    target_names = ['0', 'click','book']
    #print(classification_report(test["click_or_book"], sol, target_names=target_names))
    print(classification_report(data[y_test], sol, target_names=target_names))


# In[ ]:

from sklearn.metrics import accuracy_score
print accuracy_score(test["click_or_book"], sol)


# In[ ]:

from sklearn.cross_validation import StratifiedKFold

labels = data["click_or_book"].values

skf = StratifiedKFold(labels, n_folds=2)


for train_index, test_index in skf:
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    click_or_book_random_forest_estimation(data, X_train, X_test, y_train, y_test, predictors)



#for train_index, test_index in skf:
    #print("TRAIN:", train_index, "TEST:", test_index)
 #   X_train, X_test = train.iloc([train_index]), train.iloc([test_index])
  #  y_train, y_test = labels[train_index], labels[test_index]
   #click_or_book_random_forest_estimation(data, X_train, X_test, y_train, y_test, predictors)


# In[ ]:




# In[ ]:

actual_list = {'srch_id': test['srch_id'].tolist(),
          'actual_prop_id':test['prop_id'].tolist(),
          'values':test['click_or_book'].tolist()}

prediction_list = {'srch_id': test['srch_id'].tolist(),
              'pred_prop_id':test['prop_id'].tolist(),
              'values':joint_pred}

actual = pd.DataFrame(actual_list)

prediction = pd.DataFrame(prediction_list)


actual.sort_values(by=['srch_id', 'values'], ascending=[True, False], inplace=True)

prediction.sort_values(by=['srch_id', 'values'], axis=0, ascending=[True, False], inplace=True)

result_list = {
    'srch_id':actual['srch_id'].tolist(),
    'actual_prop_id':actual['actual_prop_id'].tolist(),
    'pred_prop_id':prediction['pred_prop_id'].tolist(),
}


result= pd.DataFrame(result_list)

result.to_csv("/home/fabricc/Desktop/prediction_file.csv",index=False)


#for a,b in zip(result.pred_prop_id,result.actual_prop_id):
 #   if(a!=b): print "not equal"

#print result


# In[ ]:

actual.to_csv("/home/fabricc/Desktop/actual.csv",index=False)


# In[ ]:

def dcg(r,k):
    r = np.asfarray(r)[:k]
    result = []
    i=1
    if len(r)>0:
        for x in r:
            #print "num"+str((2**x)-1)
            #print "den"+str(np.log2(i+1))
            result.append(((2**x)-1)/(np.log2(i+1)))
            i=i+1
        #print result
        return np.sum(result)
        
    return 0


# In[ ]:

#print actual['values']


# In[ ]:

def calculate_ndcg(srch_id_column, scores_column):
    
    actual_ndcgs = []
    current_srch_id = srch_id_column[0]
    k = 38
    i=0
    j=0
    limit = len(srch_id_column)

    for srch_id, score in zip(srch_id_column,scores_column):
       
        #print "src"+str(srch_id)
        #print current_srch_id
        #print "iteration "+str(i)
        #print "pointer "+str(current_srch_id)
        #print "srch_id "+str(srch_id)
        #i=i+1
        #if(i==10): 
            #print actual_ndcgs
            #return actual_ndcgs
        if(current_srch_id != srch_id):
            #print "ndcg value "+str(get_ndcg(scores,k))
            actual_ndcgs.append(dcg(scores_column[i:j],k))
            i=j
                
        current_srch_id = srch_id
        j=j+1
    
    actual_ndcgs.append(dcg(scores_column[i:j],k))

    #print actual_ndcgs
    return np.mean(actual_ndcgs)


# In[ ]:

ndcg_actual = calculate_ndcg(actual['srch_id'],actual['values'])


ndcg_prediction = calculate_ndcg(prediction['srch_id'],prediction['values'])



print ndcg_prediction

print ndcg_actual

print ndcg_prediction / ndcg_actual


# In[ ]:

def ndcg_calc(train_df, pred_scores):
    """
    >>ndcg_calc(train_df, pred_scores)
       train_df: pd.DataFrame with Expedia Columns: 'srch_id', 'booking_bool', 'click_bool'
       pred_scores: np.Array like vector of scores with length = num. rows in train_df
       
    Calculate Normalized Discounted Cumulative Gain for a dataset is ranked with pred_scores (higher score = higher rank).
    If 'booking_bool' == 1 then that result gets 5 points.  If 'click_bool' == 1 then that result gets 1 point (except:
    'booking_bool' = 1 implies 'click_bool' = 1, so only award 5 points total).  
    
    NDCG = DCG / IDCG
    DCG = Sum( (2 ** points - 1) / log2(rank_in_results + 1) )
    IDCG = Maximum possible DCG given the set of bookings/clicks in the training sample.
    
    """
    eval_df = train_df[['srch_id', 'booking_bool', 'click_bool']]
    eval_df['score'] = pred_scores

    logger = lambda x: math.log(x + 1, 2)
    eval_df['log_rank'] = eval_df.groupby(by = 'srch_id')['score'].rank(ascending = False).map(logger)

    book_dcg = (eval_df['booking_bool'] * 31.0 / eval_df['log_rank']).sum() #where 2 ** 5 - 1.0 = 31.0
    book_idcg = (31.0 * eval_df['booking_bool']).sum()
    
    click_dcg = (eval_df['click_bool'] * (eval_df['booking_bool'] == 0) / eval_df['log_rank']).sum()
    
    # Max number of clicks in training set is 30.
    # Calculate the 31 different contributions to IDCG that 0 to 30 clicks have
    # and put in dict {num of click: IDCG value}.
    disc = [1.0 / math.log(i + 1, 2) if i != 0 else 0 for i in range(31)]
    disc_dict = { i: np.array(disc).cumsum()[i] for i in range(31)}
    
    # Map the number of clicks to its IDCG and subtract off any clicks due to bookings
    # since these were accounted for in book_idcg.
    click_idcg = (eval_df.groupby(by = 'srch_id')['click_bool'].sum().map(disc_dict) - eval_df.groupby(by = 'srch_id')['booking_bool'].sum()).sum()

    return (book_dcg + click_dcg) / (book_idcg + click_idcg)


# In[ ]:



