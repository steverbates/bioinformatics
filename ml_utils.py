'''
Library to hold my versions of common functions and classes for machine learning analysis of bioinformatic data.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from keras.layers import Dense
from keras.models import Sequential
from math import log10, exp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier



def log10_fano(array):
	 return log10(np.nanvar(array)/np.nanmean(array)) #use variations of np.mean and np.var that leave out NaN entries



def scale_factor_calc(series): #for inputting data into models, to enable rescaling of data to maximum value for each variable; avoids division by zero in case of variables which are zero for all samples
	a = np.max(series)
	if a == 0:
		return 1
	else:
		return a



def filter_overdispersed(frame,nbins=20,top_n=500): #input should be dataframe of log-transformed expression values, with columns as gene names and each row representing a cell
	frame = frame.transpose() #for ease of calculation
	derived = pd.DataFrame(frame.apply(np.mean, axis=1),columns=['mean']) #create new data frame for derived columns
	derived['bin'] = pd.cut(derived['mean'], bins=nbins,labels=False)
	derived['log10_fano'] = frame.apply(log10_fano, axis=1)
	bin_mean, bin_std = {},{}
	for i in range(nbins):
		bin_pop = derived[derived.bin == i].log10_fano
		bin_mean[i] = np.mean(bin_pop)
		bin_std[i] = np.std(bin_pop)
	derived['dispersion'] = derived.apply(lambda x: (x.log10_fano - bin_mean[x.bin])/bin_std[x.bin],axis=1)
	derived['dispersion_rank'] = derived.dispersion.rank()
	return frame[derived.dispersion_rank<(top_n+1)].transpose() #in cases of tie at the top_nth position, assures that at least top_n genes will be found.  Transpose returned so input and returned dataframes have same axes



def logistic(t):
	return 1/(1+exp(-t))



def roc_auc(model,x=None,filepath=None,sample_weight=None,drop_intermediate=True,title='ROC Curve'): #to be called as method on binary classifiers
	if len(model.class_labels)!=2:
		return None
	if x is None: #x is test data set including true class assignments; if no x provided; calculation is made on model's training set
		y_true,y_score = model.y,model.predict_proba(model.X)['P(%s)'%(model.pos_label)]
	else:
		y_true,y_score = x[model.class_col],model.predict_proba(x)['P(%s)'%(model.pos_label)]
	fpr,tpr,thresholds = roc_curve(y_true,y_score,pos_label=model.pos_label,sample_weight=sample_weight,drop_intermediate=drop_intermediate)
	area = roc_auc_score(y_true,y_score,average=None,sample_weight=sample_weight,max_fpr=None)
	plt.clf()
	sns.set()
	plt.plot(fpr,tpr,'-g',label='AUC=%.3f'%area)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'--b')
	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate')
	plt.title(title)
	if filepath is None:
		plt.show()
		plt.close()
	else:
		plt.savefig(filepath)
		plt.close()
	return area




def heatmap(data_frame,filepath=None,categ_col=None,figsize=(6.4,4.8),title=None,center=None,dendrogram=False,method='average',metric='euclidean',z_score=None,row_linkage=None,col_linkage=None,row_colors=None): #if dendrogram True, display a hierarchically clustered map with dendrograms.  All subsequent keywords only relevant in this case.
	if center is None:
		cmap = 'mako'
	else:
		cmap = sns.diverging_palette(133,240,l=70,sep=1,n=256,center='dark')
	if categ_col is None:
		if not dendrogram:
			fig, ax = plt.subplots(figsize=figsize)
			sns.heatmap(data_frame,cmap=cmap,center=center,cbar=False,ax=ax)
			cb_ax_loc = [0.02,0.06,0.86,0.04] #in figure coordinates, left margin
			mappable = ax.collections[0] #extract information on mapping of data points to colors for generating new colorbar below
			subplots_loc = {'left':0.185,'right':0.98,'top':0.92,'bottom':0.25} #subplot limits in figure coordinates
		else:
			g = sns.clustermap(data_frame,figsize=figsize,cmap=cmap,center=center,method=method,metric=metric,z_score=z_score,row_linkage=row_linkage,col_linkage=col_linkage)
			fig, ax = g.fig, g.ax_heatmap
			cb_ax_loc = [0.02,0.01,0.86,0.04] #left margin, bottom margin, width, and height for new colorbar axis below, all in figure coordinates
			mappable = ax.collections[0] #extract information on mapping of data points to colors for generating new colorbar below
			subplots_loc = {'left':0.03,'right':0.825,'top':0.92,'bottom':0.22} #subplot limits in figure coordinates
			g.cax.set_visible(False)  #default colorbar suppressed
		ax.set_yticklabels(ax.get_ymajorticklabels(),fontsize=6)
	else: #represent categories as a column of alternating colors, independent of the main heatmap's colormap, in non-dendrogram case sorting rows so that each category is contiguous
		categories, counts = sorted(data_frame[categ_col].unique()), dict(data_frame[categ_col].value_counts())
		if -1 in categories:
			categories.remove(-1) #uncategorized samples, if they exist, to be assigned black, and removed from consideration before generating the category palette
			catmap = sns.hls_palette(len(categories),s=1,l=0.35)
			catmap, categories = [(0,0,0)]+[catmap[i//2 + (len(categories)//2)*(i%2)] for i in range(len(categories))], [-1]+categories
		else:
			catmap = sns.hls_palette(len(categories),s=1,l=0.35)
			catmap = [catmap[i//2 + (len(categories)//2)*(i%2)] for i in range(len(categories))] #the palette for category labeling was originally generated as a series of colors evenly spaced through the spectrum.  To maintain sharp boundaries even with many categories, when the transitions might otherwise become too subtle, this step rearranges the colors' order by alternately picking from the front and back halves of the spectrum, maximizing the contrast of adjacent colors
		handles = [Patch(color=catmap[i],label=str(category)+' (%i)'%counts[category]) for i,category in enumerate(categories)]
		if not dendrogram:
			fig, (ax2,ax) = plt.subplots(1,2,figsize=figsize,gridspec_kw={'width_ratios':[1,23],'wspace':0.01})
			sns.heatmap(data_frame.sort_values(categ_col).drop(columns=categ_col),cmap=cmap,center=center,yticklabels=False,cbar=False,ax=ax)
			sns.heatmap(data_frame[[categ_col]].sort_values(categ_col),cmap=catmap,xticklabels=False,yticklabels=False,cbar=False,ax=ax2)
			ax2.set_xlabel('')
			ncol = 1 + len(handles)//35
			cb_ax_loc = [0.02,0.06,0.98-0.12*ncol,0.04] #left margin, bottom margin, width, and height for new colorbar axis below, all in figure coordinates
			mappable = ax.collections[0] #extract information on mapping of data points to colors for generating new colorbar below
			subplots_loc = {'left':0.03,'right':1-0.12*ncol,'top':0.92,'bottom':0.25} #subplot limits in figure coordinates
		else:
			g = sns.clustermap(data_frame.drop(columns=categ_col),figsize=figsize,cmap=cmap,center=center,method=method,metric=metric,z_score=z_score,row_linkage=row_linkage,col_linkage=col_linkage,row_colors=data_frame[categ_col].rename('').map(dict(zip(categories,catmap))))
			fig, ax = g.fig, g.ax_heatmap
			ax.set_yticks([])
			ncol = 1 + len(handles)//35
			cb_ax_loc = [0.02,0.01,0.98-0.12*ncol,0.04] #left margin, bottom margin, width, and height for new colorbar axis below, all in figure coordinates
			subplots_loc = {'left':0.03,'right':1-0.12*ncol,'top':0.92,'bottom':0.22} #subplot limits in figure coordinates
			mappable = ax.collections[0] #extract information on mapping of data points to colors for generating new colorbar below
			g.cax.set_visible(False) #default colorbar suppressed
		fig.legend(handles=handles,bbox_to_anchor=(0.99,0.5),loc='center right',borderaxespad=0.,fontsize=6,ncol=ncol)
	cb_ax = fig.add_axes(cb_ax_loc) #create new axis solely for colorbar
	c = fig.colorbar(mappable,cax=cb_ax,orientation='horizontal') #generate colorbar
	c.ax.tick_params(labelsize=6)
	fig.subplots_adjust(**subplots_loc) #resize heatmap within figure to make room for legend
	ax.set_xlabel('')
	ax.set_ylabel('')
	ax.set_xticklabels(ax.get_xmajorticklabels(),fontsize=6)
	if title is not None:
		fig.suptitle(title)
	if filepath is None:
		plt.show(fig)
	elif dendrogram:
		g.savefig(filepath) #clustergrid object has its own savefig method which should be used rather than fig's method, to avoid cropping dendrograms
	else:
		fig.savefig(filepath)
	plt.close(fig)



class pca:
	def __init__(self,data_frame,categ_col=None,n_components=None): #data_frame rows are samples, columns are variable names; assume data already recentered/scaled as appropriate
		if categ_col is None:
			self.data = data_frame
		else:
			self.data = data_frame.drop(columns=categ_col)			
		pca = PCA(n_components=n_components,copy=False)
		if n_components is None:
			self.n_components = len(self.data.columns)
		else:
			self.n_components = n_components
		self.reduced = pd.DataFrame(pca.fit_transform(self.data),index=self.data.index,columns=['PC%i'%i for i in range(self.n_components)])
		self.components = pd.DataFrame(pca.components_,index=['PC%i'%i for i in range(self.n_components)],columns=self.data.columns)
		if categ_col is not None:
			self.reduced[categ_col] = data_frame[categ_col]
	def plot(self,filepath=None,categ_col=None,highlights=None,title=None):
		with sns.axes_style('darkgrid'):
			fig, ax = plt.subplots()
			if title is not None:
				ax.set_title(title)
		if categ_col is None:
			self.reduced.plot.scatter(x='PC0',y='PC1',s=4,ax=ax,color='k')
		else:
			if type(categ_col) == str:
				categ_col = self.reduced[categ_col]
			else:
				categ_col = pd.Series(categ_col,index=self.reduced.index)
			categories, handles = sorted(categ_col.unique()), []
			if -1 in categories:
				categories.remove(-1) #uncategorized samples, if they exist, to be assigned black, and removed from consideration before generating the category palette
				df = self.embedding[categ_col == -1]
				if highlights is None:
					df.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,color=(0,0,0),ax=ax)
					handles.append(Patch(color=(0,0,0),label=str(-1)+' (%i)'%len(df.index)))
				else:
					df.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,color=(0.8,0.8,0.8),ax=ax)
			pal = sns.hls_palette(len(categories),s=1,l=0.35)
			pal = [pal[i//2 + (len(categories)//2)*(i%2)] for i in range(len(categories))] #palette reordering not strictly necessary, but added for consistency with catmap ordering in heatmap plot method
			if highlights is None:
				for i, category in enumerate(categories):
					df = self.reduced[categ_col == category]
					df.plot.scatter(x='PC0',y='PC1',s=4,color=pal[i],ax=ax)
					handles.append(Patch(color=pal[i],label=str(category)+' (%i)'%len(df.index)))
			else:
				pal2 = sns.hls_palette(len(categories),s=0.15,l=0.8)
				pal2 = [pal2[i//2 + (len(categories)//2)*(i%2)] for i in range(len(categories))] #palette reordering not strictly necessary, but added for consistency with catmap ordering in heatmap plot method
				for i, category in enumerate(categories):
					df = self.reduced[categ_col == category]
					if category in highlights:
						df.plot.scatter(x='PC0',y='PC1',s=4,color=pal[i],ax=ax)
						handles.append(Patch(color=pal[i],label=str(category)+' (%i)'%len(df.index)))
					else:
						df.plot.scatter(x='PC0',y='PC1',s=4,color=pal2[i],ax=ax)
			ncol = 1 + len(handles)//35
			fig.legend(handles=handles,bbox_to_anchor=(0.99,0.5),loc='center right',borderaxespad=0.,fontsize=6,ncol=ncol)
			fig.subplots_adjust(left=0.07,right=(1.01-0.11*ncol))
		if filepath is None:
			plt.show()
		else:
			plt.savefig(filepath)
		plt.close(fig)



class tsne:
	def __init__(self,data_frame,categ_col=None,perplexity=30.0,early_exaggeration=12.0,learning_rate=200.0,metric='euclidean',init='pca',verbose=1): #data_frame rows are samples, columns are variable names; assume data already recentered/scaled as appropriate
		if type(data_frame) == str: #if data frame argument is a string, this is interpreted as a filepath to load a previously saved embedding from a csv, and all other arguments are overriden.
			filepath = data_frame
			self.embedding = pd.read_csv(filepath,header=[0,1],index_col=0)
			header = self.embedding.columns[0][0].split(', ') #The attributes in the header aren't strictly needed but extracting them provides a useful check on reading a csv produced by saving a previous tsne object.
			self.perplexity, self.early_exaggeration, self.learning_rate, self.metric, self.init = float(header[0].split(': ')[1]), float(header[1].split(': ')[1]), float(header[2].split(': ')[1]), header[3].split(': ')[1], header[4].split(': ')[1]
			self.embedding.columns = self.embedding.columns.droplevel()
		else:
			if categ_col is None:
				self.data = data_frame
			else:
				self.data = data_frame.drop(columns=categ_col)
			self.perplexity, self.early_exaggeration, self.learning_rate, self.metric, self.init = perplexity, early_exaggeration, learning_rate, metric, init
			self.embedding = pd.DataFrame(TSNE(perplexity=perplexity,early_exaggeration=early_exaggeration,learning_rate=learning_rate,metric=metric,init=init,verbose=verbose).fit_transform(self.data),index=self.data.index,columns=['tSNE 1','tSNE 2'])
			if categ_col is not None:
				self.embedding[categ_col] = data_frame[categ_col]
	def to_csv(self,filepath): #since tsne nondeterministic, useful to be able to save embeddings that look particularly good for later use
		header = 'perplexity: '+str(self.perplexity)+', early_exaggeration: '+str(self.early_exaggeration)+', learning_rate: '+str(self.learning_rate)+', metric: '+str(self.metric)+', init: '+str(self.init)
		pd.DataFrame(self.embedding.values,index=self.embedding.index,columns=pd.MultiIndex.from_arrays([[header]+[' ' for i in range(len(self.embedding.columns)-1)],self.embedding.columns])).to_csv(filepath) #MultiIndex trick to effectively add a header row
	def plot(self,filepath=None,categ_col=None,highlights=None,title=None):
		with sns.axes_style('darkgrid'):
			fig, ax = plt.subplots()
			if title is not None:
				ax.set_title(title)
		if categ_col is None:
			self.embedding.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,ax=ax,color='k')
		else:
			if type(categ_col) == str:
				categ_col = self.embedding[categ_col]
			else:
				categ_col = pd.Series(categ_col,index=self.embedding.index)
			categories, handles = sorted(categ_col.unique()), []
			if -1 in categories:
				categories.remove(-1) #uncategorized samples, if they exist, to be assigned black, and removed from consideration before generating the category palette
				df = self.embedding[categ_col == -1]
				if highlights is None:
					df.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,color=(0,0,0),ax=ax)
					handles.append(Patch(color=(0,0,0),label=str(-1)+' (%i)'%len(df.index)))
				else:
					df.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,color=(0.8,0.8,0.8),ax=ax)
			pal = sns.hls_palette(len(categories),s=1,l=0.35)
			pal = [pal[i//2 + (len(categories)//2)*(i%2)] for i in range(len(categories))] #palette reordering not strictly necessary, but added for consistency with catmap ordering in heatmap plot method
			if highlights is None:
				for i, category in enumerate(categories):
					df = self.embedding[categ_col == category]
					df.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,color=pal[i],ax=ax)
					handles.append(Patch(color=pal[i],label=str(category)+' (%i)'%len(df.index)))
			else:
				pal2 = sns.hls_palette(len(categories),s=0.15,l=0.8)
				pal2 = [pal2[i//2 + (len(categories)//2)*(i%2)] for i in range(len(categories))] #palette reordering not strictly necessary, but added for consistency with catmap ordering in heatmap plot method

				for i, category in enumerate(categories):
					df = self.embedding[categ_col == category]
					if category in highlights:
						df.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,color=pal[i],ax=ax)
						handles.append(Patch(color=pal[i],label=str(category)+' (%i)'%len(df.index)))
					else:
						df.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,color=pal2[i],ax=ax)
			ncol = 1 + len(handles)//35
			fig.legend(handles=handles,bbox_to_anchor=(0.99,0.5),loc='center right',borderaxespad=0.,fontsize=6,ncol=ncol)
			fig.subplots_adjust(left=0.07,right=(1.01-0.11*ncol))
		if filepath is None:
			plt.show()
		else:
			plt.savefig(filepath)
		plt.close(fig)



class log_reg_classifier:
	def __init__(self, data_frame,class_col,scalar_cols=None,penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None,solver='liblinear',max_iter=100, multi_class='ovr', verbose=0, warm_start=False,n_jobs=None,sample_weight=None,rescale=True,pos_label=None): #data_frame rows are samples, columns are variable names.  class_col and scalar_cols are used to set targets and input variables for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  pos_label can optionally be used to force the label name considered to be positive (i.e. higher probability in sigmoid output function) in two-way classification. All other parameters defaults to pass to LogisticRegression class.
		self.y = data_frame[class_col]
		if scalar_cols is None:
			self.X = data_frame.drop(columns=class_col)
			scalar_cols = self.X.columns
		else:
			self.X = data_frame[scalar_cols]
		if rescale:
			self.scale_factors = np.max(self.X,axis=0)
		else:
			self.scale_factors = self.X.apply(lambda x: 1,axis=0)
		self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs).fit(self.X/self.scale_factors,self.y,sample_weight=sample_weight)
		self.intercept = self.model.intercept_[0]
		self.coefficients = self.model.coef_[0]/self.scale_factors
		self.class_labels = self.model.classes_
		if len(self.class_labels) == 2:
			if pos_label is None or pos_label not in self.class_labels:
				self.pos_label = self.class_labels[1]
			else:
				self.pos_label = pos_label
				if self.class_labels[1] != pos_label:
					self.class_labels[0] = self.class_labels[1]
					self.class_labels[1] = pos_label
		self.scalar_cols = scalar_cols
		self.class_col = class_col
	def predict(self,x=None):
		if x is None:
			x = self.X
		return pd.DataFrame(self.model.predict(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['Predicted Class'])
	def predict_proba(self,x=None):
		if x is None:
			x = self.X
		return pd.DataFrame(self.model.predict_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['P(%s)'%s for s in self.class_labels])
	def predict_log_proba(self,x=None):
		if x is None:
			x = self.X
		return pd.DataFrame(self.model.predict_log_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['log(P(%s))'%s for s in self.class_labels])
	def confusion_matrix(self,x=None,sample_weight=None):
		if x is None:
			y_true, y_pred = self.y, self.predict(self.X)
		else:
			y_true, y_pred = x[self.class_col], self.predict(x)
		cm = confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight)
		return pd.DataFrame(cm,index=['%s True'%s for s in self.class_labels],columns=['%s Pred.'%s for s in self.class_labels])
	def roc_auc(self,x=None,filepath=None,sample_weight=None,drop_intermediate=True,title=None):
		if title is None:
			title = '%s Logistic Regression Model ROC Curve'%(self.class_col)
		return roc_auc(self,x,filepath=filepath,sample_weight=sample_weight,drop_intermediate=drop_intermediate,title=title)
	def decision_function(self,x):
		return pd.DataFrame(self.model.decision_function(x[self.scalar_cols]/self.scale_factors),index=x.index)
	def predictors(self,x=None):
		if x is None:
			x = self.X
		frame = pd.DataFrame(self.model.predict_log_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=self.class_labels)
		return frame[frame.columns[1]] - frame[frame.columns[0]]



class knn_classifier:
	def __init__(self,data_frame,class_col,scalar_cols=None,n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None,rescale=True,pos_label=None): #Supervised nearest neighbors model for classifying according to predetermined classes.  data_frame rows are samples, columns are variable names.  class_col and scalar_cols are used to set targets and input variables for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  pos_label can optionally be used to force the label name considered to be positive in two-way classification. All other parameters defaults to pass to KNeighborsClassifier class.
		self.y = data_frame[class_col]
		if scalar_cols is None:
			self.X = data_frame.drop(columns=class_col)
			scalar_cols = self.X.columns
		else:
			self.X = data_frame[scalar_cols]
		if rescale:
			self.scale_factors = np.max(self.X,axis=0)
		else:
			self.scale_factors = self.X.apply(lambda x: 1,axis=0)
		self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs).fit(self.X/self.scale_factors,self.y)
		self.scalar_cols = scalar_cols
		self.class_col = class_col
		self.class_labels = sorted(data_frame[class_col].unique())
		if len(self.class_labels) == 2:
			if pos_label is None or pos_label not in self.class_labels:
				self.pos_label = self.class_labels[1]
			else:
				self.pos_label = pos_label
				if self.class_labels[1] != pos_label:
					self.class_labels[0] = self.class_labels[1]
					self.class_labels[1] = pos_label
		self.n_neighbors = n_neighbors
	def predict(self,x=None):
		if x is None:
			x = self.X
		return pd.DataFrame(self.model.predict(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['Predicted Class'])
	def predict_proba(self,x=None):
		if x is None:
			x = self.X
		return pd.DataFrame(self.model.predict_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['P(%s)'%s for s in self.class_labels])
	def confusion_matrix(self,x=None,sample_weight=None):
		if x is None:
			y_true, y_pred = self.y, self.predict(self.X)
		else:
			y_true, y_pred = x[self.class_col], self.predict(x)
		cm = confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight)
		return pd.DataFrame(cm,index=['%s True'%s for s in self.class_labels],columns=['%s Pred.'%s for s in self.class_labels])
	def roc_auc(self,x=None,filepath=None,sample_weight=None,drop_intermediate=True,title=None):
		if title is None:
			title = '%s %s-Nearest Neighbors Model ROC Curve'%(self.class_col,str(self.n_neighbors))
		return roc_auc(self,x,filepath=filepath,sample_weight=sample_weight,drop_intermediate=drop_intermediate,title=title)



class random_forest_classifier:
	def __init__(self,data_frame,class_col,scalar_cols=None,n_estimators=100,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,bootstrap=True,oob_score=False,n_jobs=None,random_state=None,verbose=0,warm_start=False,class_weight=None,sample_weight=None,rescale=True,pos_label=None): #For classifying according to predetermined classes.  data_frame rows are samples, columns are variable names.  class_col and scalar_cols are used to set targets and input variables for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  pos_label can optionally be used to force the label name considered to be positive in two-way classification. All other parameters defaults to pass to RandomForestClassifier class.
		self.y = data_frame[class_col]
		if scalar_cols is None:
			self.X = data_frame.drop(columns=class_col)
			scalar_cols = self.X.columns
		else:
			self.X = data_frame[scalar_cols]
		if rescale:
			self.scale_factors = np.max(self.X,axis=0)
		else:
			self.scale_factors = self.X.apply(lambda x: 1,axis=0)
		self.model = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease,min_impurity_split=min_impurity_split,bootstrap=bootstrap,oob_score=oob_score,n_jobs=n_jobs,random_state=random_state,verbose=verbose,warm_start=warm_start,class_weight=class_weight).fit(self.X/self.scale_factors,self.y,sample_weight=sample_weight)
		self.scalar_cols = scalar_cols
		self.class_col = class_col
		self.class_labels = self.model.classes_
		if len(self.class_labels) == 2:
			if pos_label is None or pos_label not in self.class_labels:
				self.pos_label = self.class_labels[1]
			else:
				self.pos_label = pos_label
				if self.class_labels[1] != pos_label:
					self.class_labels[0] = self.class_labels[1]
					self.class_labels[1] = pos_label
		self.feature_importances = self.model.feature_importances_
		if oob_score:
			self.oob_score = self.model.oob_score_
			self.oob_decision_function = self.model.oob_decision_function_
	def predict(self,x=None):
		if x is None:
			x = self.X
		return pd.DataFrame(self.model.predict(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['Predicted Class'])
	def predict_proba(self,x=None):
		if x is None:
			x = self.X
		return pd.DataFrame(self.model.predict_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['P(%s)'%s for s in self.class_labels])
	def predict_log_proba(self,x=None):
		if x is None:
			x = self.X
		return pd.DataFrame(self.model.predict_log_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['log(P(%s))'%s for s in self.class_labels])
	def confusion_matrix(self,x=None,sample_weight=None):
		if x is None:
			y_true, y_pred = self.y, self.predict(self.X)
		else:
			y_true, y_pred = x[self.class_col], self.predict(x)
		cm = confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight)
		return pd.DataFrame(cm,index=['%s True'%s for s in self.class_labels],columns=['%s Pred.'%s for s in self.class_labels])
	def roc_auc(self,x=None,filepath=None,sample_weight=None,drop_intermediate=True,title=None):
		if title is None:
			title = '%s Random Forest Model ROC Curve'%(self.class_col)
		return roc_auc(self,x,filepath=filepath,sample_weight=sample_weight,drop_intermediate=drop_intermediate,title=title)



class dense_network_classifier:
	def __init__(self,data_frame,class_col,scalar_cols=None,units=[16],activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,optimizer='rmsprop',metrics=['accuracy'],epochs=10,batch_size=None,verbose=1,validation_split=0.15,validation_data=None,class_weight=None,sample_weight=None,steps_per_epoch=None,validation_steps=None,rescale=True,pos_label=None): #Neural network model for classifying according to predetermined classes.  data_frame rows are samples, columns are variable names.  class_col and scalar_cols are used to set targets and input variables for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  pos_label can optionally be used to force the label name considered to be positive (i.e. higher probality in sigmoid output function) in two-way classification. Assumption is that there are at least two densely connected layers in sequence, with either a softmax or signmoid activation in the final layer, to ouput probabilities.  The other layers are all assumed to have the same activation, with sizes set by the list of items in the units paramater. All other parameters defaults to pass to keras model and layers.
		self.y = data_frame[class_col]
		self.y_one_hot = pd.get_dummies(self.y)
		if scalar_cols is None:
			self.X = data_frame.drop(columns=class_col)
			scalar_cols = self.X.columns
		else:
			self.X = data_frame[scalar_cols]
		if rescale:
			self.scale_factors = self.X.apply(scale_factor_calc,axis=0)
		else:
			self.scale_factors = self.X.apply(lambda x: 1,axis=0)
		self.scalar_cols, self.class_col = scalar_cols, class_col
		self.class_labels = self.y_one_hot.columns
		if len(self.class_labels) == 2:
			if pos_label is None or pos_label not in self.class_labels:
				self.pos_label = self.class_labels[1]
			else:
				self.pos_label = pos_label
				if self.class_labels[1] != pos_label:
					self.class_labels[0] = self.class_labels[1]
					self.class_labels[1] = pos_label
		self.model = Sequential()
		self.model.add(Dense(units=units[0],activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,input_shape=(len(scalar_cols),)))
		if len(units) > 1:
			for u in units[1:]:
				self.model.add(Dense(units=u,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint))
		if len(self.class_labels) > 2:
			self.model.add(Dense(units=len(self.class_labels),activation='softmax'))
			loss = 'categorical_crossentropy' #to make sure label order stays the same
			self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
			self.history = self.model.fit(self.X/self.scale_factors,self.y_one_hot,epochs=epochs,batch_size=batch_size,verbose=verbose,validation_split=validation_split,validation_data=validation_data,class_weight=class_weight,sample_weight=sample_weight,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)
		else:
			self.model.add(Dense(units=1,activation='sigmoid'))
			loss = 'binary_crossentropy'
			self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
			d = {label:i for i,label in enumerate(self.class_labels)}
			self.history = self.model.fit(self.X/self.scale_factors,self.y.map(d),epochs=epochs,batch_size=batch_size,verbose=verbose,validation_split=validation_split,validation_data=validation_data,class_weight=class_weight,sample_weight=sample_weight,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)
		self.loss, self.acc, self.epochs = self.history.history['loss'], self.history.history['acc'], range(1,epochs+1)
		self.validation_split, self.validation_data, self.batch_size = validation_split, validation_data, batch_size
		if validation_split > 0.0 or validation_data is not None:
			self.val_loss, self.val_acc = self.history.history['val_loss'], self.history.history['val_acc']
	def plot_history(self,filepath=None):
		sns.set(palette='bright')
		fig,axes = plt.subplots(2,sharex='col')
		axes[1].plot(self.epochs, self.acc, 'oc', label='Training')
		axes[0].plot(self.epochs,self.loss, 'oc', label='Training')
		if self.validation_split > 0.0 or self.validation_data is not None:
			axes[1].plot(self.epochs, self.val_acc, '-g', label='Validation')
			axes[0].plot(self.epochs, self.val_loss, '-g', label='Validation')
		axes[0].legend(bbox_to_anchor=(0., 1.02),loc='lower center',ncol=2)	
		axes[0].set_ylabel('Loss')
		axes[1].set_ylabel('Accuracy')
		plt.xlabel('Epochs')
		if filepath is None:
			plt.show()
			plt.close('fig')
		else:
			plt.savefig(filepath)
			plt.close('fig')
	def predict(self,x=None,batch_size=32,verbose=1):
		if x is None:
			x = self.X
		return pd.DataFrame(self.model.predict_classes(x[self.scalar_cols]/self.scale_factors,batch_size=batch_size,verbose=verbose),index=x.index,columns=['Predicted Class']).applymap(lambda i: self.class_labels[i])
	def predict_proba(self,x=None,batch_size=None,verbose=1,steps=None):
		if x is None:
			x = self.X
		if len(self.class_labels) > 2:
			return pd.DataFrame(self.model.predict(x[self.scalar_cols]/self.scale_factors,batch_size=batch_size,verbose=verbose,steps=steps),index=x.index,columns=['P(%s)'%s for s in self.class_labels])
		else:
			p = self.model.predict(x[self.scalar_cols]/self.scale_factors,batch_size=batch_size,verbose=verbose,steps=steps).flatten()
			q = 1-p
			return pd.DataFrame({'P('+str(self.class_labels[0])+')':q,'P('+str(self.class_labels[1])+')':p},index=x.index)
	def confusion_matrix(self,x=None,sample_weight=None):
		if x is None:
			y_true, y_pred = self.y, self.predict()
		else:
			y_true, y_pred = x[self.class_col], self.predict(x)
		cm = confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight)
		return pd.DataFrame(cm,index=['%s True'%s for s in self.class_labels],columns=['%s Pred.'%s for s in self.class_labels])
	def roc_auc(self,x=None,filepath=None,sample_weight=None,drop_intermediate=True,title=None):
		if title is None:
			title = '%s Dense Neural Network Classifier ROC Curve'%(self.class_col,)
		return roc_auc(self,x,filepath=filepath,sample_weight=sample_weight,drop_intermediate=drop_intermediate,title=title)



#Function to take an unbalanced data set for a two-fold classification problem (two possible classes: positive or negative) and effectively balance it, producing a data set with the same population size for each class.  The class with the smaller population in the unbalanced data set will be unaltered in the balanced data set, while a sample of the same size will be randomly chosen without replacement from the other class.
def assemble_input_set(positives,negatives,fold=1): #Assumes dataframe inputs.  Fold parameter used to generate a set of samples for cross-validation
	n,p = len(negatives),len(positives)
	if fold==1:
		if n<p:
			x = np.concatenate((positives.sample(n=n).values,negatives.values))
			y = np.concatenate((np.ones(n),np.zeros(n)))
			order = np.arange(2*n) 
		else:
			x = np.concatenate((positives.values,negatives.sample(n=p).values))
			y = np.concatenate((np.ones(p),np.zeros(p)))
			order = np.arange(2*p)
		np.random.shuffle(order)
		return x[order,:],y[order] #shuffling positive and negative data before output; otherwise fit function would take validation split from back end i.e. all negative data
	else:
		if n<p:
			P = np.array_split(positives.sample(n=n).values,fold)
			N = np.array_split(negatives.values,fold)
		else:
			P = np.array_split(positives.values,fold)
			N = np.array_split(negatives.sample(n=p).values,fold)
		x,y = [],[]
		for i in range(len(P)):
			X = np.concatenate((P[i],N[i]))
			q = int(len(X)/2)
			Y = np.concatenate((np.ones(q),np.zeros(q)))
			order = np.arange(2*q)
			np.random.shuffle(order)
			x.append(X[order,:])
			y.append(Y[order])
		return x,y

#Untested after this point:
'''

#from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
class hier_agg_cluster:
	def __init__(self,data_frame,n_clusters=2,affinity='euclidean',memory=None,connectivity=None,compute_full_tree='auto',linkage='ward',distance_threshold=None,rescale=True):
		self.data = data_frame
		if rescale:
			self.scale_factors = self.data.apply(scale_factor_calc,axis=0)
		else:
			self.scale_factors = self.data.apply(lambda x: 1,axis=0)
		model = AgglomerativeClustering(n_clusters=n_clusters,affinity=affinity,memory=memory,connectivity=connectivity,compute_full_tree=compute_full_tree,linkage=linkage,pooling_func=pooling_func,distance_threshold=distance_threshold)
		self.data['cluster'] = model.fit_predict(self.data/self.scale_factors)
		self.n_clusters, self.n_leaves, self.n_connected_components, self.children = model.n_clusters_, model.n_leaves_, model.n_connected_components_, model.children_



class DBSCAN_cluster:
	def __init__(self,data_frame,eps=0.5,min_samples=5,metric='euclidean',metric_params=None,algorithm='auto',leaf_size=30,p=None,n_jobs=None,rescale=True):
		self.data = data_frame
		if rescale:
			self.scale_factors = self.data.apply(scale_factor_calc,axis=0)
		else:
			self.scale_factors = self.data.apply(lambda x: 1,axis=0)
		model = DBSCANneps=neps,min_samples=min_samples,metric=metric,metric_params=metric_params,algorithm=algorithm,leaf_size=leaf_size,p=p,n_jobs=n_jobs)
		self.data['cluster'] = model.fit_predict(self.data/self.scale_factors)
#		def tsne_embed(self,perplexity=30.0,learning_rate=200.0,metric='euclidean',init='pca',verbose=0):
#			self.tsne_embedding = 1


'''


'''

class shared_nn_cluster:
	def __init__(self,data_frame):
		self.data = data_frame
	if rescale:
		self.scale_factors = self.data.apply(scale_calc,axis=0)
	else:
		self.scale_factors = self.data.apply(lambda x: 1,axis=0)
'''	

#import sklearn.impute

