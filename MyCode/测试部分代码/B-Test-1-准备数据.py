
# coding: utf-8

# In[1]:


from utils.imports import *


# In[ ]:


data_path = PATH['src_test']
preded_path = PATH['model_test_pred']


# In[ ]:

#加载测试数据
patients = load_train(data_path)


# In[ ]:

#创建一个进程
Parallel(n_jobs=-1)(delayed(pred_tests)(data_path,patient,preded_path) for patient in tqdm(sorted(patients)))


# In[ ]:





# In[ ]:




