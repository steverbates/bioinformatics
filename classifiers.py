import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump,load 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score



def rescale_factors_calc(series): #for inputting data into models
	minimum,maximum = np.min(series), np.max(series)
	range_ = maximum - minimum
	if range_==0:
		return minimum,1 #to avoid division by zero when actually applying scale factors
	else:
		return minimum,range_



class random_forest_classifier:
	def __init__(self,data_frame,categ_col=None,scalar_cols=None,n_estimators=100,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,bootstrap=True,oob_score=False,n_jobs=None,random_state=None,verbose=0,warm_start=False,class_weight=None,sample_weight=None,rescale=True,pos_label=None): #For classifying according to predetermined classes.  data_frame rows are samples, columns are variable names.  categ_col and scalar_cols are used to set targets and input variables for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  pos_label can optionally be used to force the label name considered to be positive in two-way classification. All other parameters are defaults to pass to RandomForestClassifier class.
		if type(data_frame) == str: #if data frame argument is a string, this is interpreted as a filepath to load a previously saved model from a file, and all other arguments are overriden.
			saved = load(data_frame)
			for var in vars(saved).keys():
				setattr(self,var,vars(saved)[var])
		else:
			if categ_col is None:
				categ_col = data_frame.columns[-1]
			elif type (categ_col) != str: #assumption is that categ_col is part of data_frame, so need to add it if it's been provided as a separate iterable
				data_frame['categ_col'] = pd.Series(categ_col,index=data_frame.index)
				categ_col = 'categ_col'
			self.y_train = data_frame[categ_col]
			if scalar_cols is None:
				self.X_train = data_frame.drop(columns=categ_col)
				scalar_cols = self.X_train.columns
			else:
				self.X_train = data_frame[scalar_cols]
			if rescale:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(rescale_factors_calc,axis=0)),columns=['minima','ranges'])
			else:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(lambda x: (0,1),axis=0)),columns=['minima','ranges'])
			self.model = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease,min_impurity_split=min_impurity_split,bootstrap=bootstrap,oob_score=oob_score,n_jobs=n_jobs,random_state=random_state,verbose=verbose,warm_start=warm_start,class_weight=class_weight).fit((self.X_train - self.rescale_factors['minima'])/self.rescale_factors['ranges'],self.y_train,sample_weight=sample_weight)
			self.scalar_cols = scalar_cols
			self.categ_col = categ_col
			self.class_labels = self.model.classes_
			if len(self.class_labels) == 2: #Need to identify positive/negative labels to calculate ROC curve, sensitivity, specificity, false positive rate, positive predictive value
				if pos_label is None or pos_label not in self.class_labels:
					self.pos_label = self.class_labels[1]
				else:
					self.pos_label = pos_label
					if self.class_labels[1] != pos_label:
						self.class_labels[0] = self.class_labels[1]
						self.class_labels[1] = pos_label
	def predict(self,X=None):
		if X is None:
			X = self.X_test
		else:
			self.X_test = X
		self.y_pred = pd.Series(self.model.predict((X[self.scalar_cols]-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=X.index)
		try:
			delattr(self,'y_pred_proba')
		except:
			pass
		return self.y_pred
	def predict_proba(self,X=None):
		if X is None:
			X = self.X_test
		else:
			self.X_test = X
		self.y_pred_proba = pd.DataFrame(self.model.predict_proba((X[self.scalar_cols]-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=X.index,columns=['P(%s)'%s for s in self.class_labels])
		self.y_pred = self.y_pred_proba.idxmax(axis=1).map({'P(%s)'%s:s for s in self.class_labels}).rename('Predicted Class')
		return self.y_pred_proba
	def confusion_matrix(self,X=None,sample_weight=None):
		if X is not None:
			y_true, y_pred = X[self.categ_col], self.predict(X)
			self.y_test = y_true
			cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
			self.cm = cm
		else:
			cm = self.cm
		return cm
	def confusion_matrix_plot(self,X=None,savepath=None,sample_weight=None,title=None,figsize=(6.4,4.8),cmap='hot_r'):
		if title is None:
			title = '%s Random Forest Model Confusion Matrix'%(self.categ_col)
		if X is not None:
			y_true, y_pred = X[self.categ_col], self.predict(X)
			self.y_test = y_true
			cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted')) #Setting index the same as columns, as I wanted, raises an error when feeding cm to sns.heatmap, I so had to hack it like so.
			self.cm = cm
		else:
			cm = self.cm
		fig, ax = plt.subplots(figsize=figsize)
		sns.heatmap(cm,cmap=cmap,vmax=max([cm[col].nlargest(2)[1] for col in cm.columns]),annot=True, linewidths=0.02, linecolor='k', fmt='d', ax=ax)
		ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
		ax.set_title(title)
		fig.tight_layout()
		if savepath is None:
			plt.show()
		else:
			fig.savefig(savepath)
	def confusion_metrics(self,X=None,sample_weight=None):
		if X is not None:
			y_true, y_pred = X[self.categ_col], self.predict(X)
			self.y_test = y_true
			cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
			self.cm = cm
		else:
			cm = self.cm
		acc = np.trace(cm)/cm.values.sum()
		err = 1- acc
		if len(self.class_labels)==2:
			p = self.pos_label
			n = [x for x in self.class_labels if x!= p][0]
			sens = cm.loc['%s'%p,p]/cm.loc['%s'%p,:].sum()
			spec = cm.loc['%s'%n,n]/cm.loc['%s'%n,:].sum()
			ppv = cm.loc['%s'%p,p]/cm.loc[:,p].sum()
			fpr = cm.loc['%s'%n,p]/cm.loc['%s'%n,:].sum()
			return {'Accuracy':acc,'Error Rate':err,'Sensitivity':sens,'Specificity':spec,'Positive Predictive Value':ppv,'False Positive Rate':fpr}
		return {'Accuracy':acc,'Error Rate':err}
	def roc_auc(self,X=None,plot=True,savepath=None,sample_weight=None,drop_intermediate=True,title=None,figsize=(6.4,4.8)):
		if title is None:
			title = '%s Random Forest Model ROC Curve'%(self.categ_col)
		if len(self.class_labels)!=2:
			return None
		if X is None:
			y_true, y_score = self.y_test,self.predict_proba()['P(%s)'%(self.pos_label)]
		else:
			y_true, y_score = X[self.categ_col],self.predict_proba(X)['P(%s)'%(self.pos_label)]
			self.y_test = y_true
		area = roc_auc_score(y_true,y_score,average=None,sample_weight=sample_weight,max_fpr=None)
		if plot:
			fpr, tpr, thresholds = roc_curve(y_true,y_score,pos_label=self.pos_label,sample_weight=sample_weight,drop_intermediate=drop_intermediate)
			fig, ax = plt.subplots(figsize=figsize)
			sns.set()
			ax.plot(fpr,tpr,'-g',label='AUC=%.3f'%area)
			ax.legend(loc='lower right')
			ax.plot([0,1],[0,1],'--b')
			ax.set_xlabel('false positive rate')
			ax.set_ylabel('true positive rate')
			ax.set_title(title)
			fig.tight_layout()
			if savepath is None:
				plt.show()
			else:
				fig.savefig(savepath)
		self.roc_auc = area
		return area
	def save(self,savepath='random_forest_classifier.joblib'):
		dump(self,savepath)
	def delattr(self,attributes):
		for attr in pd.Series(attributes):
			delattr(self,attr)
	def clear_data(self):
		for attr in ['X_train','y_train','X_test','y_test','y_pred','y_pred_proba']:
			try:
				delattr(self,attr)
			except:
				pass



class log_reg_classifier:
	def __init__(self, data_frame,categ_col=None,scalar_cols=None,penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None,solver='liblinear',max_iter=100, multi_class='ovr', verbose=0, warm_start=False,n_jobs=None,sample_weight=None,rescale=True,pos_label=None): #data_frame rows are samples, columns are variable names.  categ_col and scalar_cols are used to set targets and input variables for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  pos_label can optionally be used to force the label name considered to be positive (i.e. higher probability in sigmoid output function) in two-way classification. All other parameters defaults to pass to LogisticRegression class.
		if type(data_frame) == str: #if data frame argument is a string, this is interpreted as a filepath to load a previously saved model from a file, and all other arguments are overriden.
			saved = load(data_frame)
			for var in vars(saved).keys():
				setattr(self,var,vars(saved)[var])
		else:
			if categ_col is None:
				categ_col = data_frame.columns[-1]
			elif type (categ_col) != str: #assumption is that categ_col is part of data_frame, so need to add it if it's been provided as a separate iterable
				data_frame['categ_col'] = pd.Series(categ_col,index=data_frame.index)
				categ_col = 'categ_col'
			self.y_train = data_frame[categ_col]
			if scalar_cols is None:
				self.X_train = data_frame.drop(columns=categ_col)
				scalar_cols = self.X_train.columns
			else:
				self.X_train = data_frame[scalar_cols]
			if rescale:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(rescale_factors_calc,axis=0)),columns=['minima','ranges'])
			else:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(lambda x: (0,1),axis=0)),columns=['minima','ranges'])
			self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs).fit((self.X_train - self.rescale_factors['minima'])/self.rescale_factors['ranges'],self.y_train,sample_weight=sample_weight)
			self.scalar_cols = scalar_cols
			self.categ_col = categ_col
			self.class_labels = self.model.classes_
			if len(self.class_labels) == 2: #Need to identify positive/negative labels to calculate ROC curve, sensitivity, specificity, false positive rate, positive predictive value
				if pos_label is None or pos_label not in self.class_labels:
					self.pos_label = self.class_labels[1]
				else:
					self.pos_label = pos_label
					if self.class_labels[1] != pos_label:
						self.class_labels[0] = self.class_labels[1]
						self.class_labels[1] = pos_label
	def predict(self,X=None):
		if X is None:
			X = self.X_test
		else:
			self.X_test = X
		self.y_pred = pd.Series(self.model.predict((X[self.scalar_cols]-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=X.index)
		try:
			delattr(self,'y_pred_proba')
		except:
			pass
		return self.y_pred
	def predict_proba(self,X=None):
		if X is None:
			X = self.X_test
		else:
			self.X_test = X
		self.y_pred_proba = pd.DataFrame(self.model.predict_proba((X[self.scalar_cols]-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=X.index,columns=['P(%s)'%s for s in self.class_labels])
		self.y_pred = self.y_pred_proba.idxmax(axis=1).map({'P(%s)'%s:s for s in self.class_labels}).rename('Predicted Class')
		return self.y_pred_proba
	def confusion_matrix(self,X=None,sample_weight=None):
		if X is not None:
			y_true, y_pred = X[self.categ_col], self.predict(X)
			self.y_test = y_true
			cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
			self.cm = cm
		else:
			cm = self.cm
		return cm
	def confusion_matrix_plot(self,X=None,savepath=None,sample_weight=None,title=None,figsize=(6.4,4.8),cmap='hot_r'):
		if title is None:
			title = '%s Logistic Regression Model Confusion Matrix'%(self.categ_col)
		if X is not None:
			y_true, y_pred = X[self.categ_col], self.predict(X)
			self.y_test = y_true
			cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted')) #Setting index the same as columns, as I wanted, raises an error when feeding cm to sns.heatmap, I so had to hack it like so.
			self.cm = cm
		else:
			cm = self.cm
		fig, ax = plt.subplots(figsize=figsize)
		sns.heatmap(cm,cmap=cmap,vmax=max([cm[col].nlargest(2)[1] for col in cm.columns]),annot=True, linewidths=0.02, linecolor='k', fmt='d', ax=ax)
		ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
		ax.set_title(title)
		fig.tight_layout()
		if savepath is None:
			plt.show()
		else:
			fig.savefig(savepath)
	def confusion_metrics(self,X=None,sample_weight=None):
		if X is not None:
			y_true, y_pred = X[self.categ_col], self.predict(X)
			self.y_test = y_true
			cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
			self.cm = cm
		else:
			cm = self.cm
		acc = np.trace(cm)/cm.values.sum()
		err = 1- acc
		if len(self.class_labels)==2:
			p = self.pos_label
			n = [x for x in self.class_labels if x!= p][0]
			sens = cm.loc['%s'%p,p]/cm.loc['%s'%p,:].sum()
			spec = cm.loc['%s'%n,n]/cm.loc['%s'%n,:].sum()
			ppv = cm.loc['%s'%p,p]/cm.loc[:,p].sum()
			fpr = cm.loc['%s'%n,p]/cm.loc['%s'%n,:].sum()
			return {'Accuracy':acc,'Error Rate':err,'Sensitivity':sens,'Specificity':spec,'Positive Predictive Value':ppv,'False Positive Rate':fpr}
		return {'Accuracy':acc,'Error Rate':err}
	def roc_auc(self,X=None,plot=True,savepath=None,sample_weight=None,drop_intermediate=True,title=None,figsize=(6.4,4.8)):
		if title is None:
			title = '%s Logistic Regression Model ROC Curve'%(self.categ_col)
		if len(self.class_labels)!=2:
			return None
		if X is None:
			y_true, y_score = self.y_test,self.predict_proba()['P(%s)'%(self.pos_label)]
		else:
			y_true, y_score = X[self.categ_col],self.predict_proba(X)['P(%s)'%(self.pos_label)]
			self.y_test = y_true
		area = roc_auc_score(y_true,y_score,average=None,sample_weight=sample_weight,max_fpr=None)
		if plot:
			fpr, tpr, thresholds = roc_curve(y_true,y_score,pos_label=self.pos_label,sample_weight=sample_weight,drop_intermediate=drop_intermediate)
			fig, ax = plt.subplots(figsize=figsize)
			sns.set()
			ax.plot(fpr,tpr,'-g',label='AUC=%.3f'%area)
			ax.legend(loc='lower right')
			ax.plot([0,1],[0,1],'--b')
			ax.set_xlabel('false positive rate')
			ax.set_ylabel('true positive rate')
			ax.set_title(title)
			fig.tight_layout()
			if savepath is None:
				plt.show()
			else:
				fig.savefig(savepath)
		self.roc_auc = area
		return area
	def save(self,savepath='log_reg_classifier.joblib'):
		dump(self,savepath)
	def delattr(self,attributes):
		for attr in pd.Series(attributes):
			delattr(self,attr)
	def clear_data(self):
		for attr in ['X_train','y_train','X_test','y_test','y_pred','y_pred_proba']:
			try:
				delattr(self,attr)
			except:
				pass