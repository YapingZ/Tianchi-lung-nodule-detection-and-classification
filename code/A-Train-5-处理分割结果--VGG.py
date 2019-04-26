
# coding: utf-8

# In[ ]:


from utils.imports import *


# In[2]:


csv_path = PATH['annotations_train']
pred_csv_path = PATH['model_train_pred']
data_path = PATH['src_train']


# # 一、所有预测结果

# In[3]:


anno_csv = pd.read_csv(csv_path + "annotations_all.csv")
pred_csv = pd.read_csv(pred_csv_path + "0_vgg_final_result.csv")

anno_csv_new = cal_recall(pred_csv,anno_csv)
pred_csv_new_temp = cal_dist(pred_csv,anno_csv)

pred_csv_new_true = pred_csv_new_temp.copy()
pred_csv_true = pred_csv_new_true[pred_csv_new_true['distmax']<16]
pred_csv_new = pred_csv_new_temp[pred_csv_new_temp['distmin']>48]


# In[4]:


num_node = len(anno_csv_new)*1.0
score_0 = anno_csv_new[anno_csv_new['ratio'] < 0.125].count()[0]/num_node
score_1 = anno_csv_new[anno_csv_new['ratio'] < 0.25].count()[0]/num_node
score_2 = anno_csv_new[anno_csv_new['ratio'] < 0.5].count()[0]/num_node
score_3 = anno_csv_new[anno_csv_new['ratio'] < 1].count()[0]/num_node
score_4 = anno_csv_new[anno_csv_new['ratio'] < 2].count()[0]/num_node
score_5 = anno_csv_new[anno_csv_new['ratio'] < 4].count()[0]/num_node
score_6 = anno_csv_new[anno_csv_new['ratio'] < 8].count()[0]/num_node

print(u"小于0.125：\t%.2f%%"  %(score_0*100))
print(u"小于0.25：\t%.2f%%"  %(score_1*100))
print(u"小于0.5：\t%.2f%%"  %(score_2*100))
print(u"小于1：\t\t%.2f%%"  %(score_3*100))
print(u"小于2：\t\t%.2f%%"  %(score_4*100))
print(u"小于4：\t\t%.2f%%"  %(score_5*100))
print(u"小于8：\t\t%.2f%%"  %(score_6*100))

score = (score_0*7+score_1*6+score_2*5+score_3*4+score_4*3+score_5*2+score_6*1)/27.
print(u"查全率：\t\t%.2f%%"  %(score*100))
print(u"查准率：\t\t%.2f%%"  %((anno_csv_new[anno_csv_new['ratio'] < 1].count()[0]*1.0/len(pred_csv_new)*1.0)*100))
print(u"负样本数量：\t%s个"  %(len(pred_csv_new)))


anno_csv_new.to_csv(pred_csv_path + "0_vgg_anno_csv_new.csv", index=False)
pred_csv_new.to_csv(pred_csv_path + "0_vgg_pred_csv_new.csv", index=False)
pred_csv_true.to_csv(pred_csv_path + "0_vgg_pred_csv_true.csv", index=False)


# # 二、直径3的开运算

# In[5]:


anno_csv = pd.read_csv(csv_path + "annotations_all.csv")
pred_csv = pd.read_csv(pred_csv_path + "1_vgg_final_result.csv")

anno_csv_new = cal_recall(pred_csv,anno_csv)
pred_csv_new_temp = cal_dist(pred_csv,anno_csv)

pred_csv_new_true = pred_csv_new_temp.copy()
pred_csv_true = pred_csv_new_true[pred_csv_new_true['distmax']<16]
pred_csv_new = pred_csv_new_temp[pred_csv_new_temp['distmin']>48]


# In[6]:


num_node = len(anno_csv_new)*1.0
score_0 = anno_csv_new[anno_csv_new['ratio'] < 0.125].count()[0]/num_node
score_1 = anno_csv_new[anno_csv_new['ratio'] < 0.25].count()[0]/num_node
score_2 = anno_csv_new[anno_csv_new['ratio'] < 0.5].count()[0]/num_node
score_3 = anno_csv_new[anno_csv_new['ratio'] < 1].count()[0]/num_node
score_4 = anno_csv_new[anno_csv_new['ratio'] < 2].count()[0]/num_node
score_5 = anno_csv_new[anno_csv_new['ratio'] < 4].count()[0]/num_node
score_6 = anno_csv_new[anno_csv_new['ratio'] < 8].count()[0]/num_node

print(u"小于0.125：\t%.2f%%"  %(score_0*100))
print(u"小于0.25：\t%.2f%%"  %(score_1*100))
print(u"小于0.5：\t%.2f%%"  %(score_2*100))
print(u"小于1：\t\t%.2f%%"  %(score_3*100))
print(u"小于2：\t\t%.2f%%"  %(score_4*100))
print(u"小于4：\t\t%.2f%%"  %(score_5*100))
print(u"小于8：\t\t%.2f%%"  %(score_6*100))

score = (score_0*7+score_1*6+score_2*5+score_3*4+score_4*3+score_5*2+score_6*1)/27.
print(u"查全率：\t\t%.2f%%"  %(score*100))
print(u"查准率：\t\t%.2f%%"  %((anno_csv_new[anno_csv_new['ratio'] < 1].count()[0]*1.0/len(pred_csv_new)*1.0)*100))
print(u"负样本数量：\t%s个"  %(len(pred_csv_new)))


anno_csv_new.to_csv(pred_csv_path + "1_vgg_anno_csv_new.csv", index=False)
pred_csv_new.to_csv(pred_csv_path + "1_vgg_pred_csv_new.csv", index=False)
pred_csv_true.to_csv(pred_csv_path + "1_vgg_pred_csv_true.csv", index=False)


# # 三、直径5的开运算

# In[7]:


anno_csv = pd.read_csv(csv_path + "annotations_all.csv")
pred_csv = pd.read_csv(pred_csv_path + "2_vgg_final_result.csv")

anno_csv_new = cal_recall(pred_csv,anno_csv)
pred_csv_new_temp = cal_dist(pred_csv,anno_csv)

pred_csv_new_true = pred_csv_new_temp.copy()
pred_csv_true = pred_csv_new_true[pred_csv_new_true['distmax']<16]
pred_csv_new = pred_csv_new_temp[pred_csv_new_temp['distmin']>48]


# In[8]:


num_node = len(anno_csv_new)*1.0
score_0 = anno_csv_new[anno_csv_new['ratio'] < 0.125].count()[0]/num_node
score_1 = anno_csv_new[anno_csv_new['ratio'] < 0.25].count()[0]/num_node
score_2 = anno_csv_new[anno_csv_new['ratio'] < 0.5].count()[0]/num_node
score_3 = anno_csv_new[anno_csv_new['ratio'] < 1].count()[0]/num_node
score_4 = anno_csv_new[anno_csv_new['ratio'] < 2].count()[0]/num_node
score_5 = anno_csv_new[anno_csv_new['ratio'] < 4].count()[0]/num_node
score_6 = anno_csv_new[anno_csv_new['ratio'] < 8].count()[0]/num_node

print(u"小于0.125：\t%.2f%%"  %(score_0*100))
print(u"小于0.25：\t%.2f%%"  %(score_1*100))
print(u"小于0.5：\t%.2f%%"  %(score_2*100))
print(u"小于1：\t\t%.2f%%"  %(score_3*100))
print(u"小于2：\t\t%.2f%%"  %(score_4*100))
print(u"小于4：\t\t%.2f%%"  %(score_5*100))
print(u"小于8：\t\t%.2f%%"  %(score_6*100))

score = (score_0*7+score_1*6+score_2*5+score_3*4+score_4*3+score_5*2+score_6*1)/27.
print(u"查全率：\t\t%.2f%%"  %(score*100))
print(u"查准率：\t\t%.2f%%"  %((anno_csv_new[anno_csv_new['ratio'] < 1].count()[0]*1.0/len(pred_csv_new)*1.0)*100))
print(u"负样本数量：\t%s个"  %(len(pred_csv_new)))


anno_csv_new.to_csv(pred_csv_path + "2_vgg_anno_csv_new.csv", index=False)
pred_csv_new.to_csv(pred_csv_path + "2_vgg_pred_csv_new.csv", index=False)
pred_csv_true.to_csv(pred_csv_path + "2_vgg_pred_csv_true.csv", index=False)


# # 四、抽样合并假结节表

# In[9]:


anno_false_0 = pd.read_csv(pred_csv_path + "0_vgg_pred_csv_new.csv")
anno_false_1 = pd.read_csv(pred_csv_path + "1_vgg_pred_csv_new.csv")
anno_false_2 = pd.read_csv(pred_csv_path + "2_vgg_pred_csv_new.csv")


# In[10]:


anno_false_0 = anno_false_0[anno_false_0.index%18 == 0]
anno_false_1 = anno_false_1[anno_false_1.index%5 == 0]


# In[11]:


pd.concat([anno_false_0,anno_false_1,anno_false_2],axis=0).to_csv(pred_csv_path + "vgg_anno_false_final.csv", index=False)


# # 五、合并真结节表

# In[12]:


anno_true_0 = pd.read_csv(pred_csv_path + "0_vgg_pred_csv_true.csv")
anno_true_1 = pd.read_csv(pred_csv_path + "1_vgg_pred_csv_true.csv")
anno_true_2 = pd.read_csv(pred_csv_path + "2_vgg_pred_csv_true.csv")


# In[13]:


pd.concat([anno_true_0,anno_true_1,anno_true_2],axis=0).to_csv(pred_csv_path + "vgg_anno_true_final.csv", index=False)


# In[ ]:




