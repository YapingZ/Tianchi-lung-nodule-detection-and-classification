#引入模块
from utils.imports import *

#加载路径
model_paths = PATH['model_paths']
data_path = PATH['model_val']

#加载分割模型
model_fenge_path=model_paths + 'final_fenge_VGG.h5'
model_fenge = load_model(model_fenge_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})


lungs = [x for x in sorted(os.listdir(data_path + 'lung/')) if x != '.DS_Store']
nods = [x for x in sorted(os.listdir(data_path + 'nodule/')) if x != '.DS_Store']

mean = 0.0
for scan in tqdm(lungs):
    patient_id = scan.split('/')[-1][:-4]
    img = cv2.imread(data_path + 'lung/' + scan, cv2.IMREAD_GRAYSCALE)
    # seg_img, overlap = helpers.get_segmented_lungs(img.copy()*255)
    mask = cv2.imread(data_path + 'nodule/' + scan[:-5] + 'm.png', cv2.IMREAD_GRAYSCALE).astype(int)

    # img = skimage.morphology.binary_opening(np.squeeze(img), np.ones([2,2]))
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    p = model_fenge.predict(img)
    mean += dice_coef_np(mask, p)
    p = np.squeeze(p)

#框架的准确率
print('VGG Benchmark:',mean/len(lungs))

#显示肺部图片
%matplotlib inline
plt.imshow(np.squeeze(img))

#显示结节mask
plt.imshow(np.squeeze(mask))

#显示分割结节
plt.imshow(np.squeeze(p))

#使用3开运算的结果
s=p.copy()
s[s==1]=int(1)
s[s!=1]=int(0)
s=np.squeeze(s)
s = skimage.morphology.binary_opening(s, np.ones([3,3]))
plt.imshow(s)