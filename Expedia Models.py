
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
import output_ndcg as on


data = pd.read_csv("/home/fabricc/Desktop/final_data/sampled_training_set.csv")
#test_set = pd.read_csv('/home/fabricc/Desktop/final_data/test_cleaned.csv')
#train_set, test_set = train_test_split(data, test_size = 0.3)
#train = data
#train_set.sort_values(by=['srch_id', 'click_or_book'], axis=0, ascending=[True, False], inplace=True)
#test_set.sort_values(by=['srch_id', 'click_or_book'], axis=0, ascending=[True, False], inplace=True)
#test_set = pd.read_csv("/home/fabricc/Desktop/final_data/test_cleaned.csv")


# In[2]:

def train(train,predictors,label,model,weight=False):
	labels = train[label].values
	features = train[predictors].values
	    
	model.fit(features,labels);


    
	if(weight):
		print "Features sorted by their score:"
		print sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), predictors),reverse=True)
	return model
    


def train_and_predict(train_set,test_set,predictors,label,model,target_names,weight=False):

	#predictors = ['prop_location_score2','price_usd','prop_location_score1','prop_log_historical_price']

	model = train(train_set,predictors,label,model,weight)

	sol = model.predict(test_set[predictors])

	actual = test_set[label]
	print(classification_report(actual, sol, target_names=target_names))
    
	ncdg = (on.calculate_dcg(test_set['srch_id'].tolist(), sol,"pred"))/ 			(on.calculate_dcg(test_set['srch_id'].tolist(), test_set["click_or_book"].tolist(),"act"))
	print ncdg
	result = {"srch_id":test_set['srch_id'].tolist(),"actual_score":test_set['click_or_book'].copy().tolist(),"pred_score":sol}
	pd.DataFrame(result).to_csv("/home/fabricc/Desktop/ndcg_dump/prediction.csv",index=False)
    
    
    
def train_and_predict_with_prob(train_set,test_set,predictors,label,model,weight=False):

	#predictors = ['prop_location_score2','price_usd','prop_location_score1','prop_log_historical_price']

	model = train(train_set,predictors,label,model,weight)

	temp = model.predict_proba(test_set[predictors])

    
    
	sol = []
	
	for x in temp:
		sol.append(x[1])
        
	result = {"srch_id":test_set['srch_id'].copy().tolist(),"prob":sol,"score_pred":test_set['click_or_book'].copy().tolist()}
	pred_df=pd.DataFrame(result)
	#print pred_df
	pred_df = pred_df.sample(frac=1)
	pred_df.sort_values(by=['srch_id', 'prob'], axis=0, ascending=[True, False], inplace=True)
    
	pred_df['actual']=test_set['click_or_book'].copy().tolist()
	#print pred_df

	pred_df.to_csv("/home/fabricc/Desktop/final_data/prediction_check2.csv")
    
    
	ncdg = (on.calculate_dcg(test_set['srch_id'].tolist(), pred_df.score_pred.tolist(),"pred"))/    (on.calculate_dcg(test_set['srch_id'].tolist(), pred_df.actual.tolist(),"act"))
	#actual = test[label]
	#print actual
	#print test['click_or_book']
	print "NDCG value: "+str(ncdg)
    
    
def train_and_predict_with_prob_real(train_set,test_set,predictors,label,model,weight=False):

	#predictors = ['prop_location_score2','price_usd','prop_location_score1','prop_log_historical_price']

	model = train(train_set,predictors,label,model,weight)

	temp = model.predict_proba(test_set[predictors])

    
    
	sol = []
	
	for x in temp:
		sol.append(x[1])
        
	result_list = {
		'SearchId':train_set['srch_id'].tolist(),
		'PropertyId':train_set['prop_id'].tolist(),
		'values':sol
		}
	result = pd.DataFrame(result_list)

	result.sort_values(by=['SearchId', 'values'], axis=0, ascending=[True, False], inplace=True)
	
	result = result[['SearchId', 'PropertyId']]

	result.to_csv("/home/fabricc/Desktop/final_data/official_pred.csv",index=False,sep=',')


# In[8]:

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)

predictors = ['site_id','prop_country_id','prop_starrating','prop_review_score',
            'prop_brand_bool','prop_location_score1','prop_location_score2',
             'prop_log_historical_price','promotion_flag',
              'srch_length_of_stay', 'srch_booking_window','srch_adults_count',
                'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
                    'srch_query_affinity_score',
                        'random_bool','month','time','usd_diff','price_usd']
#train_and_predict(train,test,predictors,"click_or_book",rfc,['0', 'click','book'],weight=True)
model=train(data,predictors,"booking_bool",rfc,weight=False)


# In[14]:

sol = []


i = 1

result_list = {
	'SearchId':[],
	'PropertyId':[],
	'values':[]
	}
    

for s in range(0,5):
	test_set = pd.read_csv("/home/fabricc/Desktop/TestCleaning/Splits/subtest_complete{0}.csv".format(i))

	temp = model.predict_proba(test_set[predictors])
	for x in temp:
		sol.append(x[1])

	i = i + 1  

	result_list['SearchId']=result_list['SearchId']+test_set['srch_id'].tolist()
	result_list['PropertyId']=result_list['PropertyId']+test_set['prop_id'].tolist()
	result_list['values']=result_list['values']+sol
	sol=[]  

result = pd.DataFrame(result_list)

result.sort_values(by=['SearchId', 'values'], axis=0, ascending=[True, False], inplace=True)

result = result[['SearchId', 'PropertyId']]

result.to_csv("/home/fabricc/Desktop/final_data/official_pred.csv",index=False,sep=',')


# In[15]:

result.shape


# In[ ]:

result_list = {
		'SearchId':data['srch_id'].tolist(),
		'PropertyId':train_set['prop_id'].tolist(),
		'values':sol
		}
result = pd.DataFrame(result_list)

result.sort_values(by=['SearchId', 'values'], axis=0, ascending=[True, False], inplace=True)

result = result[['SearchId', 'PropertyId']]

result.to_csv("/home/fabricc/Desktop/final_data/official_pred.csv",index=False,sep=',')


# In[2]:

#Data Preparation - Sampling by ID
import random


ids = data.srch_id.unique().tolist()


num_ids = len(ids)

size_sample = (num_ids * 70 ) / 100 

ids_sample = []

for i in range(0,size_sample):
    selected_id = ids[random.randrange(0,num_ids)]
    ids_sample.append(selected_id)
    ids.remove(selected_id)
    num_ids = num_ids - 1


# In[8]:

train_set = data.loc[data['srch_id'].isin(ids_sample)]
test_set = data.loc[np.logical_not(data['srch_id'].isin(ids_sample))]


# In[26]:

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(n_jobs=-1)

predictors=['site_id','prop_country_id','prop_starrating','prop_review_score',
            'prop_brand_bool','prop_location_score1','prop_location_score2',
             'prop_log_historical_price','promotion_flag',
              'srch_length_of_stay', 'srch_booking_window','srch_adults_count',
                'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
                    'srch_query_affinity_score', 
                        'random_bool','month','time','cheaper_comp_count','price_usd','usd_diff']


train_and_predict_with_prob(train_set,test_set,predictors,"booking_bool",lr)


# In[27]:

predictors = ['site_id','prop_country_id','prop_starrating','prop_review_score',
            'prop_brand_bool','prop_location_score1','prop_location_score2',
             'prop_log_historical_price','promotion_flag',
              'srch_length_of_stay', 'srch_booking_window','srch_adults_count',
                'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
                    'srch_query_affinity_score',
                        'random_bool','month','time','cheaper_comp_count','usd_diff','price_usd']

train_and_predict_with_prob(train_set,test_set,predictors,"booking_bool",rfc,weight=True)


# In[25]:

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=100, subsample=1.0, min_samples_split=500, min_samples_leaf=1, 
                                 min_weight_fraction_leaf=0.0, max_depth=5, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, 
                                 warm_start=False, presort='auto')

predictors=['prop_country_id','prop_starrating','prop_review_score',
            'prop_brand_bool','prop_location_score1','prop_location_score2',
             'prop_log_historical_price','promotion_flag',
              'srch_length_of_stay', 'srch_booking_window','srch_adults_count',
                'srch_children_count', 'srch_query_affinity_score','random_bool','cheaper_comp_count','price_usd','usd_diff']


train_and_predict_with_prob(train_set,test_set,predictors,"booking_bool",gbc,weight=True)


# In[9]:

sol = []
    
for x in temp:
    sol.append(x[1])
    
prediction_file_list = {'SearchId':test_set['srch_id'].tolist(),'PropertyId':test_set['prop_id'].tolist(),'prob':sol}
pred = pd.DataFrame(prediction_file_list).sort_values(by=['SearchId','prob'],ascending=[True,False])
pred.to_csv('/home/fabricc/Desktop/final_data/prediction.csv',Index=False)


# In[11]:

import pandas as pd

pred = pd.read_csv('/home/fabricc/Desktop/final_data/prediction.csv')


pred = pred[['SearchId','PropertyId']]
pred.to_csv('/home/fabricc/Desktop/final_data/prediction.csv',index=False,sep=',')


# from sklearn.ensemble import RandomForestClassifier
# 
# rfc = RandomForestClassifier(n_estimators=20, n_jobs=-1)
# 
# predictors = ['prop_location_score2','price_usd','prop_location_score1','prop_log_historical_price']
# 
# 
# model = train(data,predictors,"booking_bool",model)
# 
# 
# i = 1
# for s in split:
#     test = pd.read_csv("/home/fabricc/Desktop/TestCleaning/Splits/subtest{0}.csv".format(i))
#     i = i + 1
#     rfc = RandomForestClassifier(n_estimators=20, n_jobs=-1)
# 
#     predictors = ['prop_location_score2','price_usd','prop_location_score1','prop_log_historical_price']
# 
#     

# In[7]:

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50, n_jobs=-1)

predictors = ['prop_location_score2','price_usd','prop_location_score1','prop_log_historical_price']

#train_and_predict(train,test,predictors,"click_or_book",rfc,['0', 'click','book'],weight=True)
model = train(train_set,predictors,"booking_bool",rfc,weight=True)

from sklearn.externals import joblib
joblib.dump(model, 'randomforest.pkl') 

#train_and_predict_with_prob(train_set,test_set,predictors,"booking_bool",rfc,weight=True)


# In[4]:

from sklearn.externals import joblib
model = joblib.load('randomforest.pkl')


# In[6]:

temp = model.predict_proba(test_set[['prop_location_score2','price_usd','prop_location_score1','prop_log_historical_price']])


# from sklearn.ensemble import GradientBoostingClassifier
# 
# removed_columns = ['Unnamed: 0','click_or_book','year','click_bool','booking_bool','position']
# 
# predictors = [c for c in train.columns if c not in removed_columns]
# 
# 
# labels = train["click_or_book"].values
# features = train[predictors].values
# 
# 
# gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, 
#                                  min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, 
#                                  warm_start=False, presort='auto')
# 
# gbc.fit(features, labels) 
# 
# sol = gbc.predict(test[predictors])
# 
# actual = test["click_or_book"]
# target_names = ['0', 'click','book']
# print(classification_report(actual, sol, target_names=target_names))
# print "Ndcg value: "+str(on.calculate_ndcg(actual, sol, test["click_or_book"]))
# 
