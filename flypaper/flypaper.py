#Prior to running this program, htseq_logCPM_hq.tab.gz, labels_HDBSCAN.csv, libs.csv, and genes_GH146_ICIM.txt must be saved to the folder with relative path '../data/paper_data'.  These files can be found in the data folder at the github link provided in the Cell paper.  To hold the created figures, the flypaper_files folder and its subfolders neuro_cluster, dbscan_cluster, agglomerative_cluster, birch_cluster, gmm_cluster, meanshift_cluster, and kmeans_cluster must also be created.
#Meanwhile, tsn18.csv must be saved at '../data'.  This file preserves a nice-looking t-stochastic neighbor embedding result for the projection neurons, and was used to make sure the same input would be supplied to test every clustering algorithm.
from datetime import datetime
start = datetime.now()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from ml_utils import heatmap, tsne, sculpt_tree



astro_mrkrs = ['alrm','Eaat1','Gat','Gs2','Msr-110','Tre1','Cyp4g15','mfas','Obp44a']
neuro_mrkrs = ['brp','nSyb','elav','Syt1','CadN']

filepath = '../data/paper_data/htseq_logCPM_hq.tab.gz'
df_hq = pd.read_csv(filepath,index_col='symbol',sep='\t',low_memory=False)
df_hq.index.rename('gene',inplace=True)
df_hq = df_hq.transpose() #make rows cells
print('htseq_logCPM_hq data shape: ',df_hq.shape) # 2271 rows, 15524 columns ie 2271 cells, 15524 genes

libs = pd.read_csv('../data/paper_data/libs.csv',index_col='library')
print('libs shape: ',libs.shape) #2476 samples, 12 attributes (including num_cells attribute, indicating 8 samples are multi-cellular)
libs = libs.loc[df_hq.index.intersection(libs.index),:]
print('libs shape: ',libs.shape) #2271 samples, 12 attributes i.e. only cells present in df_hq.
print ('genotypes: ',libs.genotype.value_counts())#'label' column is same as 'genotype' column except that Mz19-GFP categories in latter is apparently broken down in former by development stage. From paper:
#'Cells that were labeled with neuron-specific GAL4 drivers (GH146+, Mz19+, 91G04+, and Trol+) were filtered for expression of ca- nonical neuronal genes (elav, brp, Syt1, nSyb, CadN, and mCD8GFP), retaining only those cells that expressed at least 4/6 genes at > 15 CPM. After filtering, 97.3% of GH146+ PN cells express mCD8GFP (at > 15 CPM).'
#the 91G04+ and Mz19+ categories definitely should be removed; these were used in follow-up experiments (confirming mapping of a discovered cluster to a known PN category in the former case)

labels = pd.read_csv('../data/paper_data/labels_HDBSCAN.csv',sep='\t',dtype={'name':str,'label':int},index_col='name')
labels = labels.append(libs.loc[libs.label.isin(['esglia','astrocyte']),['label']])
df_hq = df_hq.loc[labels.index]
df_hq.index.rename('cell',inplace=True)
print('df_hq shape:', df_hq.shape) #1065, 15524

df_hq_markers = df_hq.filter(items=astro_mrkrs+neuro_mrkrs,axis=1)
df_hq_markers['label'] = labels['label'].apply(lambda x: x if x in ['esglia','astrocyte'] else 'neuron') #1066, 15
df_hq_markers.loc['marker type'] = df_hq_markers.columns.map({**{x: 'neuron marker' for x in neuro_mrkrs},**{x: 'astrocyte marker' for x in astro_mrkrs}})

heatmap(df_hq_markers.transpose(),categ_row='label',categ_col='marker type',row_sort=['neuron marker','astrocyte marker','nan'],col_sort=['neuron','astrocyte','esglia','nan'],title=f'Expression Patterns of Broad Cell Type Markers Across Three Cell Populations',savepath='flypaper_files/marker_heatmap.png',figsize=(11,4.8))
d = heatmap(df_hq_markers.transpose(),categ_row='label',categ_col='marker type',col_cluster=True,row_cluster=True,return_reordered=True,method='complete',row_sort=['neuron marker','astrocyte marker','nan'],col_sort=['neuron','astrocyte','esglia','nan'],title=f'Clustering of Three Cell Populations by Broad Cell Type Markers',savepath='flypaper_files/marker_clustermap.png',figsize=(11,4.8))

ICIM_genes = pd.read_csv('../data/paper_data/genes_GH146_ICIM.txt',header=None,names=['gene'])
df_ICIM = df_hq.filter(items=ICIM_genes.gene,axis=1)
print('data shape: ',df_ICIM.shape) #1065 cells, 561 genes

heatmap(df_ICIM.transpose(),categ_row=labels['label'].apply(lambda x: x if x in ['esglia','astrocyte'] else 'neuron'),col_sort=['neuron','astrocyte','esglia','nan'],savepath='flypaper_files/ICIM_heatmap.png',figsize=(11,4.8),title='ICIM Gene Expression Patterns Across Three Cell Populations')





#Dimensional Reduction and HDBSCAN Results
em = tsne(df_ICIM,perplexity=30,early_exaggeration=4.0,learning_rate=500,metric='correlation',verbose=0)
em.plot(savepath='flypaper_files/neuro_astro_cluster.png',categ_col=labels['label'].apply(lambda x: x if x in ['esglia','astrocyte'] else 'neuron'),categ_sort=['neuron','astrocyte','esglia','nan'],title='t-Distributed Stochastic Neighbor Embedding of ICIM Gene Expression Signatures',figsize=(11,4.8))


neurons = labels[labels['label'].apply(lambda x: x not in ['esglia','astrocyte'])]

em = tsne('../data/tsne18.csv')
em.plot(savepath='flypaper_files/neuro_cluster.png',categ_col='HDBSCAN label',title='t-SNE of ICIM Gene Expression Signatures with HDBSCAN Class Labels',figsize=(11,4.8))
frames=[]
for i in range(30):
	em.plot(categ_col='HDBSCAN label',highlights=[i],title='t-SNE of ICIM Gene Expression Signatures with HDBSCAN Class Labels',savepath=f'flypaper_files/neuro_cluster/{i:02}.png',figsize=(11,4.8))
	new_frame = Image.open(f'flypaper_files/neuro_cluster/{i:02}.png')
	frames.append(new_frame)
frames[0].save('flypaper_files/neuro_cluster.gif', format='gif',append_images=frames[1:],save_all=True,duration=400, loop=0)

heatmap(df_ICIM.transpose()[neurons.index],categ_row=neurons['label'],savepath='flypaper_files/ICIM_heatmap_labeled.png',figsize=(11,4.8),title='ICIM Gene Expression Patterns Across HDBSCAN Classes')





#Other Clustering Approaches
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=2.6,min_samples=3)
dbscan_label = pd.Series(dbscan.fit_predict(em.embedding.loc[neurons.index,['tSNE 1','tSNE 2']].values),index=neurons.index).append(labels['label'].drop(labels=neurons.index))
em.plot(categ_col=dbscan_label,title='DBSCAN Class Labels',savepath='flypaper_files/dbscan_cluster.png',figsize=(11,4.8))
frames=[]
for i in range(29):
	em.plot(categ_col=dbscan_label,highlights=[i],title='DBSCAN Class Labels',savepath=f'flypaper_files/dbscan_cluster/{i:02}.png',figsize=(11,4.8))
	new_frame = Image.open(f'flypaper_files/dbscan_cluster/{i:02}.png')
	frames.append(new_frame)
frames[0].save('flypaper_files/dbscan_cluster.gif',format='gif',append_images=frames[1:],save_all=True,duration=400,loop=0)

from scipy.cluster.hierarchy import linkage
Z = linkage(em.embedding.loc[neurons.index,['tSNE 1','tSNE 2']],method='single',metric='euclidean',optimal_ordering=True)
agg_label = pd.Series(sculpt_tree(Z,29).values,index=neurons.index)
em.plot(categ_col=agg_label,title='Agglomerative Cluster Class Labels',savepath='flypaper_files/agglomerative_cluster.png',figsize=(11,4.8))
frames=[]
for i in range(29):
	em.plot(categ_col=agg_label,highlights=[i],title='Agglomerative Cluster Class Labels',savepath=f'flypaper_files/agglomerative_cluster/{i:02}.png',figsize=(11,4.8))
	new_frame = Image.open(f'flypaper_files/agglomerative_cluster/{i:02}.png')
	frames.append(new_frame)
frames[0].save('flypaper_files/agglomerative_cluster.gif', format='gif',append_images=frames[1:],save_all=True,duration=400,loop=0)

from sklearn.cluster import Birch
birch = Birch(n_clusters=23)
birch_label = pd.Series(birch.fit_predict(em.embedding.loc[neurons.index,['tSNE 1','tSNE 2']].values),index=neurons.index)
em.plot(categ_col=birch_label,title=f'BIRCH Algorithm Class Labels',savepath='flypaper_files/birch_cluster.png',figsize=(11,4.8))
frames=[]
for i in range(23):
	em.plot(categ_col=birch_label,highlights=[i],title='BIRCH Algorithm Class Labels',savepath=f'flypaper_files/birch_cluster/{i:02}.png',figsize=(11,4.8))
	new_frame = Image.open(f'flypaper_files/birch_cluster/{i:02}.png')
	frames.append(new_frame)
frames[0].save('flypaper_files/birch_cluster.gif',format='gif',append_images=frames[1:],save_all=True,duration=400,loop=0)

from sklearn.mixture import GaussianMixture
gaussian = GaussianMixture(n_components=23)
gaussian_label = pd.Series(gaussian.fit_predict(em.embedding.loc[neurons.index,['tSNE 1','tSNE 2']].values),index=neurons.index)
em.plot(categ_col=gaussian_label,title=f'Gaussian Mixture Model Class Labels',savepath='flypaper_files/gmm_cluster.png',figsize=(11,4.8))
frames=[]
for i in range(23):
	em.plot(categ_col=gaussian_label,highlights=[i],title='Gaussian Mixture Model Class Labels',savepath=f'flypaper_files/gmm_cluster/{i:02}.png',figsize=(11,4.8))
	new_frame = Image.open(f'flypaper_files/gmm_cluster/{i:02}.png')
	frames.append(new_frame)
frames[0].save('flypaper_files/gmm_cluster.gif',format='gif',append_images=frames[1:],save_all=True,duration=400,loop=0)

from sklearn.cluster import MeanShift
meanshift = MeanShift(bandwidth=5)
meanshift_label = pd.Series(meanshift.fit_predict(em.embedding.loc[neurons.index,['tSNE 1','tSNE 2']].values),index=neurons.index)
em.plot(categ_col=meanshift_label,title=f'Mean-Shift Class Labels',savepath='flypaper_files/meanshift_cluster.png',figsize=(11,4.8))
frames=[]
for i in range(27):
	em.plot(categ_col=meanshift_label,highlights=[i],title='Mean-Shift Class Labels',savepath=f'flypaper_files/meanshift_cluster/{i:02}.png',figsize=(11,4.8))
	new_frame = Image.open(f'flypaper_files/meanshift_cluster/{i:02}.png')
	frames.append(new_frame)
frames[0].save('flypaper_files/meanshift_cluster.gif', format='gif',append_images=frames[1:],save_all=True,duration=400,loop=0)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=23)
kmeans_label = pd.Series(kmeans.fit_predict(em.embedding.loc[neurons.index,['tSNE 1','tSNE 2']].values),index=neurons.index)
em.plot(categ_col=kmeans_label,title=f'K-Means Class Labels',savepath='flypaper_files/kmeans_cluster.png',figsize=(11,4.8))
frames=[]
for i in range(29):
	em.plot(categ_col=kmeans_label,highlights=[i],title='K-Means Class Labels',savepath=f'flypaper_files/kmeans_cluster/{i:02}.png',figsize=(11,4.8))
	new_frame = Image.open(f'flypaper_files/kmeans_cluster/{i:02}.png')
	frames.append(new_frame)
frames[0].save('flypaper_files/kmeans_cluster.gif',format='gif',append_images=frames[1:],save_all=True,duration=400,loop=0)


print('Time elapsed: ',datetime.now()-start)
