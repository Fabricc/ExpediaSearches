
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

train_original = pd.read_csv("/home/fabricc/Downloads/Data Mining VU data/training_set_VU_DM_2014.csv");

#train_original = pd.read_csv("/home/fabricc/Downloads/Data Mining VU data/test_set_VU_DM_2014.csv")
#train_original = pd.read_csv("/home/fabricc/Desktop/sample_coding_purposes.csv");

#train = train_original.sample(frac=0.01)
#train = train_original


# In[ ]:

train = train_original
#Winsorizing of price usd
import scipy.stats
import numpy as np

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
#train.head
plt.figure()
train['price_usd'].plot(kind='box')


train['price_usd']=scipy.stats.mstats.winsorize(train['price_usd'], limits=0.005)


plt.figure()
train['price_usd'].plot(kind='box')
train.to_csv("/home/fabricc/Desktop/final_data/sampled_training_set.csv",Index=False)


# In[ ]:

train_original.sort_values(by=['srch_id','booking_bool', 'click_bool'], axis=0, ascending=[True,False, False], inplace=True)
#train_original.sort_values(by=['srch_id'], axis=0, ascending=[True], inplace=True)


# In[ ]:

#Data Preparation - Missing Values
train['prop_review_score'].fillna(train['prop_review_score'].min(), inplace=True)

train['orig_destination_distance'].fillna(train['orig_destination_distance'].median(), inplace=True)

train['prop_location_score2'].fillna(train['prop_location_score2'].min(), inplace=True)

min_srch_query_affinity_score = train['srch_query_affinity_score'].dropna().min()

train['srch_query_affinity_score'].fillna(0, inplace=True)


# In[ ]:

#Scaling
to_scale = train[['prop_review_score','prop_location_score1','prop_starrating']]

to_scale = (to_scale - to_scale.min()) / (to_scale.max() - to_scale.min())

train[['prop_review_score','prop_location_score1','prop_starrating']] = to_scale

#Standardization 

to_std = train['price_usd']
to_std = (to_std - to_std.mean())/(to_std.std())
train['price_usd'] = to_std


# In[ ]:

train.drop('gross_bookings_usd', axis=1, inplace=True)
train.drop('position', axis=1, inplace=True)


# In[4]:

ratio = 5
temp_ratio = 0
picked = []
current_srch_id = -1

for item in train.iterrows():
   
        current_srch_id = item[1].srch_id
        
        if(item[1].booking_bool):
            temp_ratio = ratio
            picked.append(1)
            continue
        
        if (item[1].click_bool ):
            picked.append(1)
        elif(temp_ratio > 0):
            picked.append(1)
            temp_ratio = temp_ratio -1
        else:
            picked.append(0)
        
train['picked']=picked
train = train[(train.picked==1)]


# In[5]:

train.drop('picked', axis=1, inplace=True) 


# In[14]:

import pandas as pd
import numpy as np

train = pd.read_csv("/home/fabricc/Desktop/final_data/sampled_training_set.csv")

train.to_csv("/home/fabricc/Desktop/final_data/sampled_training_set.csv",index=False)


# In[ ]:


#Data Preparation - Sampling by ID
#import random


#ids = bookings.srch_id.unique().tolist()


#num_ids = len(ids)

#size_sample = (num_ids * 40 ) / 100 

#ids_sample = []

#for i in range(0,size_sample):
 #   selected_id = ids[random.randrange(0,num_ids)]
#  ids_sample.append(selected_id)
   # ids.remove(selected_id)
    # num_ids = num_ids - 1
    
    
#train = train_original.loc[train_original['srch_id'].isin(ids_sample)]


# In[2]:

#Data Preparation - Feature Extraction 1
#Diff Star Rating

visitor_hist_starrating = train['visitor_hist_starrating']
prop_starrating = train['prop_starrating']

starrating_diff_list = []

for history, actual in zip(visitor_hist_starrating, prop_starrating):
    if(np.isnan(history)): 
        starrating_diff_list.append(np.nan)
    else:
        starrating_diff_list.append(abs(history - actual))

starrating_diff = pd.Series(starrating_diff_list, index = visitor_hist_starrating.index.values)

train['starrating_diff']=starrating_diff

#Diff gross booking usd

visitor_hist_adr_usd = train['visitor_hist_adr_usd']
price_usd = train['price_usd']

usd_diff_list = []

for history, actual in zip(visitor_hist_adr_usd, price_usd):
    if(np.isnan(history) or np.isnan(actual)): 
        usd_diff_list.append(np.nan)
    else:
        usd_diff_list.append(abs(history - actual))

usd_diff = pd.Series(usd_diff_list, index = visitor_hist_adr_usd.index.values)

train['usd_diff']=usd_diff

train.drop('visitor_hist_starrating', axis=1, inplace=True)
train.drop('visitor_hist_adr_usd', axis=1, inplace=True)


# In[3]:

#Data Preparation - Feature Extraction 1
#Date time decomposition
import datetime

#print train['date_time'][0]

breakfast_time = datetime.time(6,0,0)
lunch_time = datetime.time(12,0,0)
dinner_time = datetime.time(18,0,0)

week_day = []
month_day = []
month = []
year = []
time = []


for date_item in train['date_time'].iteritems():
    date = datetime.datetime.strptime(date_item[1], '%Y-%m-%d %H:%M:%S')
    week_day.append(date.weekday())
    month_day.append(date.day)
    month.append(date.month)
    year.append(date.year)
    t = date.time()
    if(breakfast_time > t): 
        time_range = 3 #night
    elif (lunch_time > t): 
        time_range = 0 #morning
    elif (dinner_time > t): 
        time_range = 1 #afternoon
    else:
        time_range = 2 #evening
    time.append(time_range)
    
d={'week_of_the_day':week_day,'month_day':month_day,'month':month,'year':year,'time':time}
f = pd.DataFrame(data=d, index=train['date_time'].index.values)
train = train.join(f)

train.drop('date_time', axis=1, inplace=True)


# In[5]:

#Merging Click and Booking 
click_book = []

for click,book in zip(train["click_bool"],train["booking_bool"]):
    if(book): 
        click_book.append(5)
    elif (click):
        click_book.append(1)
    else: click_book.append(0)
  
column = pd.Series(click_book, index = train["click_bool"].index.values)
train['click_or_book']= column


# In[6]:

#Data Preparation 
#Competitors attributes shrinking

cheaper_comp_count_list = []

for index, row in train.iterrows():
    count = 0
    count_nan = 0
    for x in range(1,9):
        comp_rate = "comp{0}_rate".format(x)
        comp_inv = "comp{0}_inv".format(x) 
        if (not np.isnan(row[comp_rate]) and not row[comp_inv]):
            if(row[comp_inv]==0 and row[comp_rate]==1): count=count+1
        else: count_nan = count_nan +1
    if(count_nan < 8): 
        cheaper_comp_count_list.append(count)
    else: cheaper_comp_count_list.append(np.nan)
    
train['cheaper_comp_count'] = pd.Series(cheaper_comp_count_list, index = train.index.values)


# In[7]:

#Filling missing data in derived attributes
train['cheaper_comp_count'].fillna(-1, inplace=True)
train['starrating_diff'].fillna(-1, inplace=True)
train['usd_diff'].fillna(-1, inplace=True)


# In[8]:

#train.drop('year', axis=1, inplace=True)
#train.drop('prop_id', axis=1, inplace=True)
#train.drop('click_bool', axis=1, inplace=True)
#train.drop('booking_bool', axis=1, inplace=True)


for x in range(1,9):
    train.drop("comp{0}_rate".format(x), axis=1, inplace=True)
    train.drop("comp{0}_inv".format(x), axis=1, inplace=True)
    train.drop("comp{0}_rate_percent_diff".format(x), axis=1, inplace=True)


# In[9]:

train.to_csv("/home/fabricc/Desktop/final_data/sampled_training_set.csv",index=False)


# In[ ]:

split = np.array_split(train, 5)

i = 1
for s in split:
    s.to_csv("/home/fabricc/Desktop/TestCleaning/Splits/subtest{0}.csv".format(i))
    i = i + 1


# In[ ]:

import pandas as pd
import numpy as np

i = 1
for s in range(0,5):
    train = pd.read_csv("/home/fabricc/Desktop/TestCleaning/Splits/subtest{0}.csv".format(i))
    print train.columns
    i = i + 1
    break


# In[ ]:

import pandas as pd
import numpy as np

i = 1
for s in range(0,5):
	print "start processing subset {0}".format(i)
	train = pd.read_csv("/home/fabricc/Desktop/TestCleaning/Splits/subtest{0}.csv".format(i))
	
    

    #Data Preparation 
#Competitors attributes shrinking

	cheaper_comp_count_list = []

	for index, row in train.iterrows():
		count = 0
		count_nan = 0
		for x in range(1,9):
			comp_rate = "comp{0}_rate".format(x)
			comp_inv = "comp{0}_inv".format(x) 
		if (not np.isnan(row[comp_rate]) and not row[comp_inv]):
			if(row[comp_inv]==0 and row[comp_rate]==1): count=count+1
			else: count_nan = count_nan +1
		if(count_nan < 8): 
			cheaper_comp_count_list.append(count)
		else: cheaper_comp_count_list.append(np.nan)
	    
	train['cheaper_comp_count'] = pd.Series(cheaper_comp_count_list, index = train.index.values)

	train.drop('date_time', axis=1, inplace=True)
	train.drop('visitor_hist_starrating', axis=1, inplace=True)
	train.drop('visitor_hist_adr_usd', axis=1, inplace=True)
	#train.drop('year', axis=1, inplace=True)
	#train.drop('prop_id', axis=1, inplace=True)
	#train.drop('click_bool', axis=1, inplace=True)
	#train.drop('booking_bool', axis=1, inplace=True)
	train.drop('gross_bookings_usd', axis=1, inplace=True)
	train.drop('position', axis=1, inplace=True)
	train.drop('Unnamed: 0', axis=1, inplace=True)
	train.drop('Unnamed: 0.1', axis=1, inplace=True)
	train.drop('Unnamed: 0.1.1', axis=1, inplace=True)



	for x in range(1,9):
		train.drop("comp{0}_rate".format(x), axis=1, inplace=True)
		train.drop("comp{0}_inv".format(x), axis=1, inplace=True)
		train.drop("comp{0}_rate_percent_diff".format(x), axis=1, inplace=True)
    #Filling missing data in derived attributes
	train['cheaper_comp_count'].fillna(-1, inplace=True)
	train['starrating_diff'].fillna(-1, inplace=True)
	train['usd_diff'].fillna(-1, inplace=True)

	train.to_csv("/home/fabricc/Desktop/TestCleaning/Splits/subtest_complete{0}.csv".format(i),index=False)
	print "stop processing subset {0}".format(i)
	i = i + 1


# In[ ]:

import pandas as pd
import numpy as np

i = 1
for s in range(0,5):
    train = pd.read_csv("/home/fabricc/Desktop/TestCleaning/Splits/subtest{0}.csv".format(i))
    print train.columns
    i = i + 1
    break

