
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import splits_operations as so
train_original = pd.read_csv("/home/fabricc/Desktop/Splits/Sample/WinsorizedTestSet.csv")


#train_original = pd.read_csv("/home/fabricc/Downloads/Data Mining VU data/test_set_VU_DM_2014.csv")



#train = train_original.sample(frac=0.01)
train = train_original

site_ids = train['site_id'].unique()
total = site_ids.size


# In[2]:

for i in range(1,total+1):
    print 'start processing id '+str(i)
    #split = so.preprocessing(train[(train['site_id'])==i])
    train[(train['site_id'])==i].to_csv("/home/fabricc/Desktop/Splits/test_splits/test_data_site{0}.csv".format(i))
    print 'stop processing id '+str(i)
    #del split
    


# In[1]:

import pandas as pd
import numpy as np
import splits_operations as so

for i in range(1,35):
    print 'start processing id '+str(i)
    split = pd.read_csv("/home/fabricc/Desktop/Splits/test_splits/test_data_site{0}.csv".format(i))
    split = so.preprocessing(split[(split['site_id'])==i])
    split.to_csv("/home/fabricc/Desktop/Splits/test_splits/test_data_site{0}.csv".format(i),index=False)
    del split
    print 'stop processing id '+str(i)


# In[ ]:



