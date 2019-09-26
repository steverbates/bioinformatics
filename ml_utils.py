'''
Library to hold my versions of workhorse functions and classes for machine learning analysis of bioinformatic data.
'''
import pandas as pd
import numpy as np
import random
from math import log10, exp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns



def log_fano(array):
	 return log10(np.nanvar(array)/np.nanmean(array)) #use variations of np.mean and np.var that leave out NaN entries

def filter_overdispersed(frame,nbins=20,top_n=500): #input should be dataframe of log-transformed expression values, with columns as gene names and each row representing a cell
	frame = frame.transpose() #for ease of calculation
	derived = pd.DataFrame(frame.apply(np.mean, axis=1),columns=['mean']) #create new data frame for derived columns
	derived['bin'] = pd.cut(derived['mean'], bins=nbins,labels=False)
	derived['log_fano'] = frame.apply(log_fano, axis=1)
	bin_mean, bin_std = {},{}
	for i in range(nbins):
		bin_pop = derived[derived.bin == i].log_fano
		bin_mean[i] = np.mean(bin_pop)
		bin_std[i] = np.std(bin_pop)
	derived['dispersion'] = derived.apply(lambda x: (x.log_fano - bin_mean[x.bin])/bin_std[x.bin],axis=1)
	derived['dispersion_rank'] = derived.dispersion.rank()
	return frame[derived.dispersion_rank<(top_n+1)].transpose() #in cases of tie at the top_nth position, assures that at least top_n genes will be found.  Transpose returned so input and returned dataframes have same axes


def logistic(t):
	return 1/(1+exp(-t))


def roc_auc(model,x=None,filepath=None,sample_weight=None,drop_intermediate=True,title='ROC Curve'): #to be called as method on classification model functions
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
	else:
		plt.savefig(filepath)
	return area

class pca:
	def __init__(self, data_frame,n_components=None,categ_col=None): #data_frame rows are samples, columns are variable names; assume data already recentered/scaled as appropriate
		self.data = data_frame
		pca = PCA(n_components=n_components,copy=False)
		if n_components is None:
			self.n_components = len(self.data.columns)
		else:
			self.n_components = n_components
		self.reduced = pd.DataFrame(pca.fit_transform(self.data),index=self.data.index,columns=['PC%i'%i for i in range(self.n_components)])
		self.components = pd.DataFrame(pca.components_,index=['PC%i'%i for i in range(self.n_components)],columns=self.data.columns)
	def plot(self,filepath=None):
		plt.clf()
		sns.set()
		self.reduced.plot(x='PC0',y='PC1',kind='scatter')
		if filepath is None:
			plt.show()
		else:
			plt.savefig(filepath)

class tsne:
	def __init__(self,data_frame,categ_col=None): #data_frame rows are samples, columns are variable names
		if categ_col is None:
			self.data = data_frame
		elif categ_col in data_frame.columns:
			self.categ_col = data_frame[categ_col]
			self.data = data_frame.drop(columns=categ_col)
	def embed(self,perplexity=30.0,learning_rate=200.0,metric='euclidean',init='pca',verbose=0): 
		self.perplexity = perplexity
		self.learning_rate = learning_rate
		self.metric =metric
		self.init =init
		embedded = TSNE(perplexity=perplexity,learning_rate=learning_rate,metric=metric,init=init,verbose=verbose).fit_transform(self.data)
		self.embedding = pd.DataFrame(embedded,index=self.data.index,columns=['tSNE 1','tSNE 2'])
	def plot(self,filepath=None):
		plt.clf()
		sns.set()
		self.embedding.plot(x='tSNE 1',y='tSNE 2',kind='scatter')
		if filepath is None:
			plt.show()
		else:
			plt.savefig(filepath)




class log_reg_model:
	def __init__(self, data_frame,class_col=None,scalar_cols=None,penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None,solver='liblinear',max_iter=100, multi_class='ovr', verbose=0, warm_start=False,n_jobs=None,sample_weight=None,rescale=True): #data_frame rows are samples, columns are variable names.  class_col and scalar_cols are used to set input variables and targets for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  All other parameters defaults to pass to LogisticRegression class.
		self.y = data_frame[class_col]
		if scalar_cols is None:
			self.X = data_frame.drop(columns=class_col)
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
		self.pos_label = self.class_labels[1]
		self.scalar_cols = scalar_cols
		self.class_col = class_col
	def predict_proba(self,x):
		return pd.DataFrame(self.model.predict_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['P(%s)'%s for s in self.class_labels])
	def predict_log_proba(self,x):
		return pd.DataFrame(self.model.predict_log_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['log(P(%s))'%s for s in self.class_labels])
	def predictors(self,x):
		frame = pd.DataFrame(self.model.predict_log_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=self.class_labels)
		return frame[frame.columns[1]] - frame[frame.columns[0]]
	def predict(self,x):
		return pd.DataFrame(self.model.predict(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['Predicted Class'])
	def confusion_matrix(self,x,sample_weight=None):
		cm = confusion_matrix(x[self.class_col],self.predict(x),labels=self.class_labels,sample_weight=sample_weight)
		return pd.DataFrame(cm,index=['%s True'%s for s in self.class_labels],columns=['%s Pred.'%s for s in self.class_labels])
	def decision_function(self,x):
		return pd.DataFrame(self.model.decision_function(x[self.scalar_cols]/self.scale_factors),index=x.index)
	def roc_auc(self,x=None,filepath=None,sample_weight=None,drop_intermediate=True,title=None):
		if title is None:
			title = '%s Logistic Regression Model ROC Curve'%(self.class_col)
		return roc_auc(self,x,filepath=filepath,sample_weight=sample_weight,drop_intermediate=drop_intermediate,title=title)


class knn_model:
	def __init__(self,data_frame,class_col=None,scalar_cols=None,n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None,rescale=True): #Supervised nearest neighbors model for classifying according to predetermined classes.  data_frame rows are samples, columns are variable names.  class_col and scalar_cols are used to set input variables and targets for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  All other parameters defaults to pass to KNeighborsClassifier class.
		self.y = data_frame[class_col]
		if scalar_cols is None:
			self.X = data_frame.drop(columns=class_col)
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
		self.pos_label = self.class_labels[1]
		self.n_neighbors = n_neighbors
	def predict_proba(self,x):
		return pd.DataFrame(self.model.predict_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['P(%s)'%s for s in self.class_labels])
	def predict(self,x):
		return pd.DataFrame(self.model.predict(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['Predicted Class'])
	def confusion_matrix(self,x,sample_weight=None):
		cm = confusion_matrix(x[self.class_col],self.predict(x),labels=self.class_labels,sample_weight=sample_weight)
		return pd.DataFrame(cm,index=['%s True'%s for s in self.class_labels],columns=['%s Pred.'%s for s in self.class_labels])
	def roc_auc(self,x=None,filepath=None,sample_weight=None,drop_intermediate=True,title=None):
		if title is None:
			title = '%s %s-Nearest Neighbors Model ROC Curve'%(self.class_col,str(self.n_neighbors))
		return roc_auc(self,x,filepath=filepath,sample_weight=sample_weight,drop_intermediate=drop_intermediate,title=title)


class random_forest_model:
	def __init__(self,data_frame,class_col=None,scalar_cols=None,n_estimators=100,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,bootstrap=True,oob_score=False,n_jobs=None,random_state=None,verbose=0,warm_start=False,class_weight=None,sample_weight=None,rescale=True): #data_frame rows are samples, columns are variable names.  class_col and scalar_cols are used to set input variables and targets for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  All other parameters defaults to pass to RandomForestClassifier class.
		self.y = data_frame[class_col]
		if scalar_cols is None:
			self.X = data_frame.drop(columns=class_col)
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
		self.pos_label = self.class_labels[1]
		self.feature_importances = self.model.feature_importances_
		if oob_score:
			self.oob_score = self.model.oob_score_
			self.oob_decision_function = self.model.oob_decision_function_
	def predict(self,x):
		return pd.DataFrame(self.model.predict(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['Predicted Class'])
	def predict_proba(self,x):
		return pd.DataFrame(self.model.predict_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['P(%s)'%s for s in self.class_labels])
	def predict_log_proba(self,x):
		return pd.DataFrame(self.model.predict_log_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['log(P(%s))'%s for s in self.class_labels])
	def confusion_matrix(self,x,sample_weight=None):
		cm = confusion_matrix(x[self.class_col],self.predict(x),labels=self.class_labels,sample_weight=sample_weight)
		return pd.DataFrame(cm,index=['%s True'%s for s in self.class_labels],columns=['%s Pred.'%s for s in self.class_labels])
	def roc_auc(self,x=None,filepath=None,sample_weight=None,drop_intermediate=True,title=None):
		if title is None:
			title = '%s Random Forest Model ROC Curve'%(self.class_col)
		return roc_auc(self,x,filepath=filepath,sample_weight=sample_weight,drop_intermediate=drop_intermediate,title=title)




#A helper function to take an unbalanced data set for a two-fold classification problem (two possible classes: positive or negative) and effectively balance it, producing a data set with the same population size for each class.  The class with the smaller population in the unbalanced data set will be unaltered in the balanced data set, while a sample of the same size will be randomly chosen without replacement from the other class.
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


#sns.heatmap