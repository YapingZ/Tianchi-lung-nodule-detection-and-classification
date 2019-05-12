import pandas as pd
import os
import numpy as np
csv_files =['1final_test_result_i.csv','1final_test_result_d.csv','1final_test_result_v.csv']
root ='/devdata1/ding/data/TianChi/ali/all_1final'
w = np.array([0.5,0.3,0.2])
csv_objects = [pd.read_csv(os.path.join(root,csv)) for csv in csv_files]
prob = [np.array(csv['probability']) for csv in csv_objects]
prob = np.stack(prob,axis=1)
weighted_sum = np.sum(prob*w.reshape(1,3),axis=1)

csv = csv_objects[0]
csv['probability']=weighted_sum
csv.to_csv(os.path.join(root,'val_cal_2.csv'),index=False)