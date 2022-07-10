#%%Import bibliotek
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random 
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

#%%Wczytywanie plikow


train_data_raw = pd.read_csv(r'samsung_HAR\samsung_train.txt',delim_whitespace=True,header=None)

train_labels = pd.read_csv(r'samsung_HAR\samsung_train_labels.txt',delim_whitespace=True,header=None)

test_data_raw = pd.read_csv(r'samsung_HAR\samsung_test.txt',delim_whitespace=True,header=None)

test_labels = pd.read_csv(r'samsung_HAR\samsung_test_labels.txt',delim_whitespace=True, header=None)

headers = pd.read_csv(r'samsung_HAR\features.txt',delim_whitespace=True,header=None)


'''
The following files are available for the train and test data. Their descriptions are equivalent. 

- 'train/subject_train.txt': Each row identifies the subject who performed the activity for each window sample. Its range is from 1 to 30. 

- 'train/Inertial Signals/total_acc_x_train.txt': The acceleration signal from the smartphone accelerometer X axis in standard gravity units 'g'. Every row shows a 128 element vector. The same description applies for the 'total_acc_x_train.txt' and 'total_acc_z_train.txt' files for the Y and Z axis. 

- 'train/Inertial Signals/body_acc_x_train.txt': The body acceleration signal obtained by subtracting the gravity from the total acceleration. 

- 'train/Inertial Signals/body_gyro_x_train.txt': The angular velocity vector measured by the gyroscope for each window sample. The units are radians/second. 

'''
#%% Dadawanie etykiet 

train_data_raw.columns=[headers[1]]
test_data_raw.columns=[headers[1]]

#%% opis kategori



#%% Brakujące wartisci

Temp = pd.DataFrame(train_data_raw.isnull().sum())
Temp.columns = ['Sum']
print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Sum'] > 0])) )

#%% Normalizacja

scaler = StandardScaler()
train_data_raw = scaler.fit_transform(train_data_raw)

#%% otymalna liczba klastró

ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(train_data_raw)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()

#%%
def k_means(n_clust, data_frame, true_labels):
    """
    Function k_means applies k-means clustering alrorithm on dataset and prints the crosstab of cluster and actual labels 
    and clustering performance parameters.
    
    Input:
    n_clust - number of clusters (k value)
    data_frame - dataset we want to cluster
    true_labels - original labels
    
    Output:
    1 - crosstab of cluster and actual labels
    2 - performance table
    """
    k_means = KMeans(n_clusters = n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)
    display(ct)
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
      %(k_means.inertia_,
      homogeneity_score(true_labels, y_clust),
      completeness_score(true_labels, y_clust),
      v_measure_score(true_labels, y_clust),
      adjusted_rand_score(true_labels, y_clust),
      adjusted_mutual_info_score(true_labels, y_clust),
      silhouette_score(data_frame, y_clust, metric='euclidean')))

    
#%% k=2
from sklearn.cluster import KMeans

clf = KMeans(n_clusters=2, random_state=120, n_init=30)

clf.fit(train_data_raw)
clf_labels = clf.labels_

df = pd.DataFrame({'clust_label': clf_labels, 'orig_label': train_labels[0]})
ct = pd.crosstab(df['clust_label'], df['orig_label'])
labels_pred = clf.predict(train_data_raw)
display(ct)

print('Wyniki dla k=2')
print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
      %(clf.inertia_,
      homogeneity_score(train_labels[0], labels_pred),
      completeness_score(train_labels[0], labels_pred),
      v_measure_score(train_labels[0], labels_pred),
      adjusted_rand_score(train_labels[0], labels_pred),
      adjusted_mutual_info_score(train_labels[0], labels_pred),
      silhouette_score(train_data_raw, labels_pred, metric='euclidean')))


#%% k=6

clf_6 = KMeans(n_clusters=6, random_state=120, n_init=30)

clf_6.fit(train_data_raw)
clf_6_labels = clf_6.labels_

df_6 = pd.DataFrame({'clust_label': clf_6_labels, 'orig_label': train_labels[0]})
ct_6 = pd.crosstab(df_6['clust_label'], df_6['orig_label'])
labels_pred_6 = clf_6.predict(train_data_raw)
display(ct_6)

print('Wyniki dla k=6')
print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
      %(clf_6.inertia_,
      homogeneity_score(train_labels[0], labels_pred_6),
      completeness_score(train_labels[0], labels_pred_6),
      v_measure_score(train_labels[0], labels_pred_6),
      adjusted_rand_score(train_labels[0], labels_pred_6),
      adjusted_mutual_info_score(train_labels[0], labels_pred_6),
      silhouette_score(train_data_raw, labels_pred_6, metric='euclidean')))

#%% Dimensionality Reduction

pca = PCA(random_state=123)
pca.fit(train_data_raw)
features = range(pca.n_components_)

plt.figure(figsize=(8,4))
plt.bar(features[:15], pca.explained_variance_[:15], color='lightskyblue')
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features[:15])
plt.show()

#%%
def pca_transform(n_comp):
    pca = PCA(n_components=n_comp, random_state=123)
    global Data_reduced
    Data_reduced = pca.fit_transform(train_data_raw)
    print('Shape of the new Data df: ' + str(Data_reduced.shape))
    
    
#%%
pca_transform(n_comp=1)


clf = KMeans(n_clusters=2, random_state=120, n_init=30)

clf.fit(Data_reduced)
clf_labels = clf.labels_

df = pd.DataFrame({'clust_label': clf_labels, 'orig_label': train_labels[0]})
ct = pd.crosstab(df['clust_label'], df['orig_label'])
labels_pred = clf.predict(Data_reduced)
display(ct)


print('Wyniki dla k=2 po Dimensionality Reduction:')
print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
      %(clf.inertia_,
      homogeneity_score(train_labels[0], labels_pred),
      completeness_score(train_labels[0], labels_pred),
      v_measure_score(train_labels[0], labels_pred),
      adjusted_rand_score(train_labels[0], labels_pred),
      adjusted_mutual_info_score(train_labels[0], labels_pred),
      silhouette_score(train_data_raw, labels_pred, metric='euclidean')))

    

