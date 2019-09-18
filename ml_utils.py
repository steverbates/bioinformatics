'''
Library to hold my versions of workhorse functions and classes for machine learning analysis of bioinformatic data.
'''
import pandas as pd
import numpy as np
import random
from math import log10, exp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
import matplotlib.pyplot as plt
import seaborn as sns

'''
A helper function to take an unbalanced data set for a two-fold classification problem (two possible classes: positive or negative) and effectively balance it, producing a data set with the same population size for each class.  The class with the smaller population in the unbalanced data set will be unaltered in the balanced data set, while a sample of the same size will be randomly chosen without replacement from the other class.
'''
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


def log_fano(array):
	 return log10(np.nanvar(array)/np.nanmean(array)) #use variations of np.mean and np.var that leave out NaN entries

def logistic(t):
	return 1/(1+exp(-t))

def logit(t):
	return math.log(t/(1-t))


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
		self.embedding.plot(x='tSNE 1',y='tSNE 2',kind='scatter')
		if filepath is None:
			plt.show()
		else:
			plt.savefig(filepath)

class log_reg_model:
	def __init__(self, data_frame,class_col=None,scalar_cols=None,penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None,solver='liblinear',max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=None): #data_frame rows are samples, columns are variable names; assume data already recentered/scaled as appropriate
		self.y = data_frame[class_col]
		if scalar_cols is None:
			self.X = data_frame.drop(columns=class_col)
		else:
			self.X = data_frame[scalar_cols]
		self.scale_factors = np.max(self.X,axis=0)
		self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs).fit(self.X/self.scale_factors,self.y.values)
		self.intercept = self.model.intercept_[0]
		self.coefficients = self.model.coef_[0]/self.scale_factors
		self.class_labels = self.model.classes_
		self.scalar_cols = scalar_cols
	def predict_proba(self,x):
		return pd.DataFrame(self.model.predict_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['P(%s)'%s for s in self.class_labels])
	def predict_log_proba(self,x):
		return pd.DataFrame(self.model.predict_log_proba(x[self.scalar_cols]/self.scale_factors),index=x.index,columns=['log(P(%s))'%s for s in self.class_labels])
