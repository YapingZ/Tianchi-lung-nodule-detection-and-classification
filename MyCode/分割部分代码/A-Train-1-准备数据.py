from utils.imports import *

data_path = PATH['src_train']
csv_path = PATH['annotations_train']

pic_path = PATH['pic_train']
train_lung_path = PATH['model_train_lung']
train_nodule_path = PATH['model_train_nodule']

patients = load_train(data_path)
df_node = pd.read_csv(csv_path+"annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(patients, file_name))
df_node = df_node.dropna()

Parallel(n_jobs=-1)(delayed(create_samples)(data_path,df_node,patient,pic_path) for patient in tqdm(sorted(patients)))

aa = os.listdir(pic_path)

for i in aa:
    if '_i' in i:
        shutil.copy(pic_path+i,train_lung_path)

for i in aa:
    if '_m' in i:
        shutil.copy(pic_path+i,train_nodule_path)

data_path = PATH['src_val']
csv_path = PATH['annotations_val']

pic_path = PATH['pic_val']
val_lung_path = PATH['model_val_lung']
val_nodule_path = PATH['model_val_nodule']

patients = load_train(data_path)
df_node = pd.read_csv(csv_path+"annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(patients, file_name))
df_node = df_node.dropna()

Parallel(n_jobs=-1)(delayed(create_samples)(data_path,df_node,patient,pic_path) for patient in tqdm(sorted(patients)))

aa = os.listdir(pic_path)

for i in aa:
    if '_i' in i:
        shutil.copy(pic_path+i,val_lung_path)

for i in aa:
    if '_m' in i:
        shutil.copy(pic_path+i,val_nodule_path)

data_path = PATH['src_train']
preded_path = PATH['model_train_pred']

patients = load_train(data_path)

Parallel(n_jobs=-1)(delayed(pred_tests)(data_path,df_node,patient,preded_path) for patient in tqdm(sorted(patients)))

data_path = PATH['src_val']
preded_path = PATH['model_train_pred']

patients = load_train(data_path)

Parallel(n_jobs=-1)(delayed(pred_tests)(data_path,df_node,patient,preded_path) for patient in tqdm(sorted(patients)))

