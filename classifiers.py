import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump,load 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.svm import LinearSVC, SVC



def rescale_factors_calc(series): #for inputting data into models
	minimum,maximum = np.min(series), np.max(series)
	range_ = maximum - minimum
	if range_==0:
		return minimum,1 #to avoid division by zero when actually applying scale factors
	else:
		return minimum,range_



class random_forest_classifier:
	def __init__(self,data_frame,categ_col=None,scalar_cols=None,n_estimators=100,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,bootstrap=True,oob_score=False,n_jobs=None,random_state=None,verbose=0,warm_start=False,class_weight=None,sample_weight=None,rescale=True,pos_label=None): #For classifying according to predetermined classes.  data_frame rows are samples, columns are variable names.  categ_col and scalar_cols are used to set targets and input variables for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  pos_label can optionally be used to force the label name considered to be positive in two-way classification. All other parameters are defaults to pass to RandomForestClassifier class.
		if type(data_frame) != str:
			if categ_col is None:
				self.y_train, data_frame = data_frame[data_frame.columns[-1]], data_frame.drop(columns=data_frame.columns[-1])
			elif type(categ_col) == str:
				self.y_train, data_frame = data_frame[categ_col], data_frame.drop(columns=categ_col)
			else:
				self.y_train = pd.Series(categ_col)
			if scalar_cols is None:
				scalar_cols = data_frame.columns
			self.X_train, self.scalar_cols = data_frame[scalar_cols], scalar_cols
			if rescale:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(rescale_factors_calc,axis=0)),columns=['minima','ranges'])
			else:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(lambda x: (0,1),axis=0)),columns=['minima','ranges'])
			self.model = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease,min_impurity_split=min_impurity_split,bootstrap=bootstrap,oob_score=oob_score,n_jobs=n_jobs,random_state=random_state,verbose=verbose,warm_start=warm_start,class_weight=class_weight).fit((self.X_train - self.rescale_factors['minima'])/self.rescale_factors['ranges'],self.y_train,sample_weight=sample_weight)
			self.class_labels = self.model.classes_
			if len(self.class_labels) == 2: #Need to identify positive/negative labels to calculate ROC curve, sensitivity, specificity, false positive rate, positive predictive value
				if pos_label is None or pos_label not in self.class_labels:
					self.pos_label = self.class_labels[1]
				else:
					self.pos_label = pos_label
					if self.class_labels[1] != pos_label:
						self.class_labels[0] = self.class_labels[1]
						self.class_labels[1] = pos_label
		else:  #if data frame argument is a string, this is interpreted as a filepath to load a previously saved model from a file, and all other arguments are overriden.
			saved = load(data_frame)
			for var in vars(saved).keys():
				setattr(self,var,vars(saved)[var])
	def predict(self,X=None):
		if X is not None:
			self.X_test = X[self.scalar_cols]
		self.y_pred = pd.Series(self.model.predict((self.X_test-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=self.X_test.index)
		try:
			delattr(self,'y_pred_proba')
		except:
			pass
		return self.y_pred
	def predict_proba(self,X=None):
		if X is not None:
			self.X_test = X[self.scalar_cols]
		self.y_pred_proba = pd.DataFrame(self.model.predict_proba((self.X_test-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=self.X_test.index,columns=['P(%s)'%s for s in self.class_labels])
		self.y_pred = self.y_pred_proba.idxmax(axis=1).map({'P(%s)'%s:s for s in self.class_labels}).rename('Predicted Class')
		return self.y_pred_proba
	def confusion_matrix(self,X=None,y_true=None,sample_weight=None):
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
		return self.cm
	def confusion_matrix_plot(self,X=None,y_true=None,savepath=None,sample_weight=None,title=None,figsize=(6.4,4.8),cmap='hot_r'):
		if title is None:
			title = 'Random Forest Model Confusion Matrix'
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted')) #Setting index the same as columns, as I wanted, raises an error when feeding cm to sns.heatmap, I so had to hack it like so.
		fig, ax = plt.subplots(figsize=figsize)
		sns.heatmap(self.cm,cmap=cmap,vmax=max([self.cm[col].nlargest(2)[1] for col in self.cm.columns]),annot=True, linewidths=0.02, linecolor='k', fmt='d', ax=ax)
		ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
		ax.set_title(title)
		fig.tight_layout()
		if savepath is None:
			plt.show()
		else:
			fig.savefig(savepath)
	def confusion_metrics(self,X=None,y_true=None,sample_weight=None):
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
		acc = np.trace(self.cm)/self.cm.values.sum()
		err = 1- acc
		if len(self.class_labels)==2:
			p = self.pos_label
			n = [x for x in self.class_labels if x!= p][0]
			sens = self.cm.loc['%s'%p,p]/self.cm.loc['%s'%p,:].sum()
			spec = self.cm.loc['%s'%n,n]/self.cm.loc['%s'%n,:].sum()
			ppv = self.cm.loc['%s'%p,p]/self.cm.loc[:,p].sum()
			fpr = self.cm.loc['%s'%n,p]/self.cm.loc['%s'%n,:].sum()
			return {'Accuracy':acc,'Error Rate':err,'Sensitivity':sens,'Specificity':spec,'Positive Predictive Value':ppv,'False Positive Rate':fpr}
		return {'Accuracy':acc,'Error Rate':err}
	def roc_auc(self,X=None,y_true=None,plot=True,savepath=None,sample_weight=None,drop_intermediate=True,title=None,figsize=(6.4,4.8)):
		if title is None:
			title = 'Random Forest Model ROC Curve'
		if len(self.class_labels)!=2:
			return None
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_score = X[y_true], self.predict_proba(X.drop(columns=y_true))['P(%s)'%(self.pos_label)]
					y_true = self.y_test
				else:
					self.y_test, y_score = y_true, self.predict_proba(X)['P(%s)'%(self.pos_label)]	
			else:
				self.y_test, y_score = X[self.y_train.name], self.predict_proba(X.drop(columns=self.y_train.name))['P(%s)'%(self.pos_label)]
				y_true = self.y_test
		else:
			y_true, y_score = self.y_test, self.predict_proba()['P(%s)'%(self.pos_label)]
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



class grad_boost_classifier:
	def __init__(self, data_frame,categ_col=None,scalar_cols=None,loss='deviance',learning_rate=0.1,n_estimators=100,subsample=1.0,criterion='friedman_mse',min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_depth=3,min_impurity_decrease=0.0,min_impurity_split=None,init=None,random_state=None,max_features=None,verbose=0,max_leaf_nodes=None,warm_start=False,presort='auto',validation_fraction=0.1,n_iter_no_change=None,tol=0.0001,sample_weight=None,rescale=True,pos_label=None): #data_frame rows are samples, columns are variable names.  categ_col and scalar_cols are used to set targets and input variables for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  pos_label can optionally be used to force the label name considered to be positive (i.e. higher probability in sigmoid output function) in two-way classification. All other parameters defaults to pass to GradientBoostingClassifier class.
		if type(data_frame) != str:
			if categ_col is None:
				self.y_train, data_frame = data_frame[data_frame.columns[-1]], data_frame.drop(columns=data_frame.columns[-1])
			elif type(categ_col) == str:
				self.y_train, data_frame = data_frame[categ_col], data_frame.drop(columns=categ_col)
			else:
				self.y_train = pd.Series(categ_col)
			if scalar_cols is None:
				scalar_cols = data_frame.columns
			self.X_train, self.scalar_cols = data_frame[scalar_cols], scalar_cols
			if rescale:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(rescale_factors_calc,axis=0)),columns=['minima','ranges'])
			else:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(lambda x: (0,1),axis=0)),columns=['minima','ranges'])
			self.model = GradientBoostingClassifier (loss=loss,learning_rate=learning_rate,n_estimators=n_estimators,subsample=subsample,criterion=criterion,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_depth=max_depth,min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split,init=init,random_state=random_state,max_features=max_features,verbose=verbose,max_leaf_nodes=max_leaf_nodes,warm_start=warm_start,presort=presort,validation_fraction=validation_fraction,n_iter_no_change=n_iter_no_change,tol=tol).fit((self.X_train - self.rescale_factors['minima'])/self.rescale_factors['ranges'],self.y_train,sample_weight=sample_weight)
			self.class_labels = self.model.classes_
			if len(self.class_labels) == 2: #Need to identify positive/negative labels to calculate ROC curve, sensitivity, specificity, false positive rate, positive predictive value
				if pos_label is None or pos_label not in self.class_labels:
					self.pos_label = self.class_labels[1]
				else:
					self.pos_label = pos_label
					if self.class_labels[1] != pos_label:
						self.class_labels[0] = self.class_labels[1]
						self.class_labels[1] = pos_label
		else:  #if data frame argument is a string, this is interpreted as a filepath to load a previously saved model from a file, and all other arguments are overriden.
			saved = load(data_frame)
			for var in vars(saved).keys():
				setattr(self,var,vars(saved)[var])
	def predict(self,X=None):
		if X is not None:
			self.X_test = X[self.scalar_cols]
		self.y_pred = pd.Series(self.model.predict((self.X_test-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=self.X_test.index)
		try:
			delattr(self,'y_pred_proba')
		except:
			pass
		return self.y_pred
	def predict_proba(self,X=None):
		if X is not None:
			self.X_test = X[self.scalar_cols]
		self.y_pred_proba = pd.DataFrame(self.model.predict_proba((self.X_test-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=self.X_test.index,columns=['P(%s)'%s for s in self.class_labels])
		self.y_pred = self.y_pred_proba.idxmax(axis=1).map({'P(%s)'%s:s for s in self.class_labels}).rename('Predicted Class')
		return self.y_pred_proba
	def confusion_matrix(self,X=None,y_true=None,sample_weight=None):
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
		return self.cm
	def confusion_matrix_plot(self,X=None,y_true=None,savepath=None,sample_weight=None,title=None,figsize=(6.4,4.8),cmap='hot_r'):
		if title is None:
			title = 'Gradient Boosted Tree Model Confusion Matrix'
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted')) #Setting index the same as columns, as I wanted, raises an error when feeding cm to sns.heatmap, I so had to hack it like so.
		fig, ax = plt.subplots(figsize=figsize)
		sns.heatmap(self.cm,cmap=cmap,vmax=max([self.cm[col].nlargest(2)[1] for col in self.cm.columns]),annot=True, linewidths=0.02, linecolor='k', fmt='d', ax=ax)
		ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
		ax.set_title(title)
		fig.tight_layout()
		if savepath is None:
			plt.show()
		else:
			fig.savefig(savepath)
	def confusion_metrics(self,X=None,y_true=None,sample_weight=None):
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
		acc = np.trace(self.cm)/self.cm.values.sum()
		err = 1- acc
		if len(self.class_labels)==2:
			p = self.pos_label
			n = [x for x in self.class_labels if x!= p][0]
			sens = self.cm.loc['%s'%p,p]/self.cm.loc['%s'%p,:].sum()
			spec = self.cm.loc['%s'%n,n]/self.cm.loc['%s'%n,:].sum()
			ppv = self.cm.loc['%s'%p,p]/self.cm.loc[:,p].sum()
			fpr = self.cm.loc['%s'%n,p]/self.cm.loc['%s'%n,:].sum()
			return {'Accuracy':acc,'Error Rate':err,'Sensitivity':sens,'Specificity':spec,'Positive Predictive Value':ppv,'False Positive Rate':fpr}
		return {'Accuracy':acc,'Error Rate':err}
	def roc_auc(self,X=None,y_true=None,plot=True,savepath=None,sample_weight=None,drop_intermediate=True,title=None,figsize=(6.4,4.8)):
		if title is None:
			title = 'Gradient Boosted Tree Model ROC Curve'
		if len(self.class_labels)!=2:
			return None
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_score = X[y_true], self.predict_proba(X.drop(columns=y_true))['P(%s)'%(self.pos_label)]
					y_true = self.y_test
				else:
					self.y_test, y_score = y_true, self.predict_proba(X)['P(%s)'%(self.pos_label)]	
			else:
				self.y_test, y_score = X[self.y_train.name], self.predict_proba(X.drop(columns=self.y_train.name))['P(%s)'%(self.pos_label)]
				y_true = self.y_test
		else:
			y_true, y_score = self.y_test, self.predict_proba()['P(%s)'%(self.pos_label)]
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
		return area
	def save(self,savepath='grad_boost_classifier.joblib'):
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
		if type(data_frame) != str:
			if categ_col is None:
				self.y_train, data_frame = data_frame[data_frame.columns[-1]], data_frame.drop(columns=data_frame.columns[-1])
			elif type(categ_col) == str:
				self.y_train, data_frame = data_frame[categ_col], data_frame.drop(columns=categ_col)
			else:
				self.y_train = pd.Series(categ_col)
			if scalar_cols is None:
				scalar_cols = data_frame.columns
			self.X_train, self.scalar_cols = data_frame[scalar_cols], scalar_cols
			if rescale:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(rescale_factors_calc,axis=0)),columns=['minima','ranges'])
			else:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(lambda x: (0,1),axis=0)),columns=['minima','ranges'])
			self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs).fit((self.X_train - self.rescale_factors['minima'])/self.rescale_factors['ranges'],self.y_train,sample_weight=sample_weight)
			self.class_labels = self.model.classes_
			if len(self.class_labels) == 2: #Need to identify positive/negative labels to calculate ROC curve, sensitivity, specificity, false positive rate, positive predictive value
				if pos_label is None or pos_label not in self.class_labels:
					self.pos_label = self.class_labels[1]
				else:
					self.pos_label = pos_label
					if self.class_labels[1] != pos_label:
						self.class_labels[0] = self.class_labels[1]
						self.class_labels[1] = pos_label
		else:  #if data frame argument is a string, this is interpreted as a filepath to load a previously saved model from a file, and all other arguments are overriden.
			saved = load(data_frame)
			for var in vars(saved).keys():
				setattr(self,var,vars(saved)[var])
	def predict(self,X=None):
		if X is not None:
			self.X_test = X[self.scalar_cols]
		self.y_pred = pd.Series(self.model.predict((self.X_test-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=self.X_test.index)
		try:
			delattr(self,'y_pred_proba')
		except:
			pass
		return self.y_pred
	def predict_proba(self,X=None):
		if X is not None:
			self.X_test = X[self.scalar_cols]
		self.y_pred_proba = pd.DataFrame(self.model.predict_proba((self.X_test-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=self.X_test.index,columns=['P(%s)'%s for s in self.class_labels])
		self.y_pred = self.y_pred_proba.idxmax(axis=1).map({'P(%s)'%s:s for s in self.class_labels}).rename('Predicted Class')
		return self.y_pred_proba
	def confusion_matrix(self,X=None,y_true=None,sample_weight=None):
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
		return self.cm
	def confusion_matrix_plot(self,X=None,y_true=None,savepath=None,sample_weight=None,title=None,figsize=(6.4,4.8),cmap='hot_r'):
		if title is None:
			title = 'Logistic Regression Model Confusion Matrix'
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted')) #Setting index the same as columns, as I wanted, raises an error when feeding cm to sns.heatmap, I so had to hack it like so.
		fig, ax = plt.subplots(figsize=figsize)
		sns.heatmap(self.cm,cmap=cmap,vmax=max([self.cm[col].nlargest(2)[1] for col in self.cm.columns]),annot=True, linewidths=0.02, linecolor='k', fmt='d', ax=ax)
		ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
		ax.set_title(title)
		fig.tight_layout()
		if savepath is None:
			plt.show()
		else:
			fig.savefig(savepath)
	def confusion_metrics(self,X=None,y_true=None,sample_weight=None):
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
		acc = np.trace(self.cm)/self.cm.values.sum()
		err = 1- acc
		if len(self.class_labels)==2:
			p = self.pos_label
			n = [x for x in self.class_labels if x!= p][0]
			sens = self.cm.loc['%s'%p,p]/self.cm.loc['%s'%p,:].sum()
			spec = self.cm.loc['%s'%n,n]/self.cm.loc['%s'%n,:].sum()
			ppv = self.cm.loc['%s'%p,p]/self.cm.loc[:,p].sum()
			fpr = self.cm.loc['%s'%n,p]/self.cm.loc['%s'%n,:].sum()
			return {'Accuracy':acc,'Error Rate':err,'Sensitivity':sens,'Specificity':spec,'Positive Predictive Value':ppv,'False Positive Rate':fpr}
		return {'Accuracy':acc,'Error Rate':err}
	def roc_auc(self,X=None,y_true=None,plot=True,savepath=None,sample_weight=None,drop_intermediate=True,title=None,figsize=(6.4,4.8)):
		if title is None:
			title = 'Logistic Regression Model ROC Curve'
		if len(self.class_labels)!=2:
			return None
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_score = X[y_true], self.predict_proba(X.drop(columns=y_true))['P(%s)'%(self.pos_label)]
					y_true = self.y_test
				else:
					self.y_test, y_score = y_true, self.predict_proba(X)['P(%s)'%(self.pos_label)]	
			else:
				self.y_test, y_score = X[self.y_train.name], self.predict_proba(X.drop(columns=self.y_train.name))['P(%s)'%(self.pos_label)]
				y_true = self.y_test
		else:
			y_true, y_score = self.y_test, self.predict_proba()['P(%s)'%(self.pos_label)]
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



class ridge_reg_classifier:
	def __init__(self, data_frame,categ_col=None,scalar_cols=None,alpha=1.0,fit_intercept=True, normalize=False, copy_X=True, max_iter=None,tol=0.001,class_weight=None, solver='auto',random_state=None,sample_weight=None,rescale=True,pos_label=None): #data_frame rows are samples, columns are variable names.  categ_col and scalar_cols are used to set targets and input variables for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  pos_label can optionally be used to force the label name considered to be positive (i.e. higher probability in sigmoid output function) in two-way classification. All other parameters defaults to pass to RidgeClassifier class.
		if type(data_frame) != str:
			if categ_col is None:
				self.y_train, data_frame = data_frame[data_frame.columns[-1]], data_frame.drop(columns=data_frame.columns[-1])
			elif type(categ_col) == str:
				self.y_train, data_frame = data_frame[categ_col], data_frame.drop(columns=categ_col)
			else:
				self.y_train = pd.Series(categ_col)
			if scalar_cols is None:
				scalar_cols = data_frame.columns
			self.X_train, self.scalar_cols = data_frame[scalar_cols], scalar_cols
			if rescale:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(rescale_factors_calc,axis=0)),columns=['minima','ranges'])
			else:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(lambda x: (0,1),axis=0)),columns=['minima','ranges'])
			self.model = RidgeClassifier(alpha=alpha,fit_intercept=fit_intercept,normalize=normalize,copy_X=copy_X,max_iter=max_iter,tol=tol,class_weight=class_weight,solver=solver,random_state=random_state).fit((self.X_train - self.rescale_factors['minima'])/self.rescale_factors['ranges'],self.y_train,sample_weight=sample_weight)
			self.class_labels = self.model.classes_
			if len(self.class_labels) == 2: #Need to identify positive/negative labels to calculate ROC curve, sensitivity, specificity, false positive rate, positive predictive value
				if pos_label is None or pos_label not in self.class_labels:
					self.pos_label = self.class_labels[1]
				else:
					self.pos_label = pos_label
					if self.class_labels[1] != pos_label:
						self.class_labels[0] = self.class_labels[1]
						self.class_labels[1] = pos_label
		else:  #if data frame argument is a string, this is interpreted as a filepath to load a previously saved model from a file, and all other arguments are overriden.
			saved = load(data_frame)
			for var in vars(saved).keys():
				setattr(self,var,vars(saved)[var])
	def predict(self,X=None):
		if X is not None:
			self.X_test = X[self.scalar_cols]
		self.y_pred = pd.Series(self.model.predict((self.X_test-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=self.X_test.index)
		try:
			delattr(self,'y_pred_proba')
		except:
			pass
		return self.y_pred
	def confusion_matrix(self,X=None,y_true=None,sample_weight=None):
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
		return self.cm
	def confusion_matrix_plot(self,X=None,y_true=None,savepath=None,sample_weight=None,title=None,figsize=(6.4,4.8),cmap='hot_r'):
		if title is None:
			title = 'Ridge Regression Model Confusion Matrix'
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted')) #Setting index the same as columns, as I wanted, raises an error when feeding cm to sns.heatmap, I so had to hack it like so.
		fig, ax = plt.subplots(figsize=figsize)
		sns.heatmap(self.cm,cmap=cmap,vmax=max([self.cm[col].nlargest(2)[1] for col in self.cm.columns]),annot=True, linewidths=0.02, linecolor='k', fmt='d', ax=ax)
		ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
		ax.set_title(title)
		fig.tight_layout()
		if savepath is None:
			plt.show()
		else:
			fig.savefig(savepath)
	def confusion_metrics(self,X=None,y_true=None,sample_weight=None):
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
		acc = np.trace(self.cm)/self.cm.values.sum()
		err = 1- acc
		if len(self.class_labels)==2:
			p = self.pos_label
			n = [x for x in self.class_labels if x!= p][0]
			sens = self.cm.loc['%s'%p,p]/self.cm.loc['%s'%p,:].sum()
			spec = self.cm.loc['%s'%n,n]/self.cm.loc['%s'%n,:].sum()
			ppv = self.cm.loc['%s'%p,p]/self.cm.loc[:,p].sum()
			fpr = self.cm.loc['%s'%n,p]/self.cm.loc['%s'%n,:].sum()
			return {'Accuracy':acc,'Error Rate':err,'Sensitivity':sens,'Specificity':spec,'Positive Predictive Value':ppv,'False Positive Rate':fpr}
		return {'Accuracy':acc,'Error Rate':err}
	def roc_auc(self,X=None,y_true=None,plot=True,savepath=None,sample_weight=None,drop_intermediate=True,title=None,figsize=(6.4,4.8)):
		if title is None:
			title = 'Ridge Regression Model ROC Curve'
		if len(self.class_labels)!=2:
			return None
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_score = X[y_true], self.model.decision_function(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_score = y_true, self.model.decision_function(X)
			else:
				self.y_test, y_score = X[self.y_train.name], self.model.decision_function(X.drop(columns=self.y_train.name))
				y_true = self.y_test
		else:
			y_true, y_score = self.y_test, self.model.decision_function(self.X_test)
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
		return area
	def save(self,savepath='ridge_reg_classifier.joblib'):
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




class svm_classifier:
	def __init__(self,data_frame,categ_col=None,scalar_cols=None,linear=True,penalty='l2',loss='squared_hinge',dual=True,multi_class='ovr',fit_intercept=True, intercept_scaling=1,C=1.0,kernel='rbf',degree=3,gamma='auto_deprecated',coef0=0.0,shrinking=True,tol=0.001,cache_size=200,class_weight=None,verbose=False,max_iter=-1,decision_function_shape='ovr',random_state=None,sample_weight=None,rescale=True,pos_label=None): #data_frame rows are samples, columns are variable names.  categ_col and scalar_cols are used to set targets and input variables for model, and rescale used to indicate that each input variable will be rescaled as a fraction of its maximum value; set to False if data has already been conditioned.  pos_label can optionally be used to force the label name considered to be positive (i.e. higher probability in sigmoid output function) in two-way classification. All other parameters defaults to pass to SVC or LinearSVC class.
		if type(data_frame) != str:
			if categ_col is None:
				self.y_train, data_frame = data_frame[data_frame.columns[-1]], data_frame.drop(columns=data_frame.columns[-1])
			elif type(categ_col) == str:
				self.y_train, data_frame = data_frame[categ_col], data_frame.drop(columns=categ_col)
			else:
				self.y_train = pd.Series(categ_col)
			if scalar_cols is None:
				scalar_cols = data_frame.columns
			self.X_train, self.scalar_cols = data_frame[scalar_cols], scalar_cols
			if rescale:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(rescale_factors_calc,axis=0)),columns=['minima','ranges'])
			else:
				self.rescale_factors = pd.DataFrame(list(self.X_train.apply(lambda x: (0,1),axis=0)),columns=['minima','ranges'])
			if linear:
				if max_iter == -1:
					max_iter = 1000
				self.model = LinearSVC(penalty=penalty,loss=loss,dual=dual,tol=tol,C=C,multi_class=multi_class,fit_intercept=fit_intercept,intercept_scaling=intercept_scaling,class_weight=class_weight,verbose=verbose,random_state=random_state,max_iter=max_iter).fit((self.X_train - self.rescale_factors['minima'])/self.rescale_factors['ranges'],self.y_train,sample_weight=sample_weight)
			else:
				self.model = SVC(C=C,kernel=kernel,degree=degree,gamma=gamma,coef0=coef0,shrinking=shrinking,probability=True,tol=tol,cache_size=cache_size,class_weight=class_weight,verbose=verbose,max_iter=max_iter,decision_function_shape=decision_function_shape,random_state=random_state).fit((self.X_train - self.rescale_factors['minima'])/self.rescale_factors['ranges'],self.y_train,sample_weight=sample_weight)
			self.class_labels = self.model.classes_
			if len(self.class_labels) == 2: #Need to identify positive/negative labels to calculate ROC curve, sensitivity, specificity, false positive rate, positive predictive value
				if pos_label is None or pos_label not in self.class_labels:
					self.pos_label = self.class_labels[1]
				else:
					self.pos_label = pos_label
					if self.class_labels[1] != pos_label:
						self.class_labels[0] = self.class_labels[1]
						self.class_labels[1] = pos_label
		else:  #if data frame argument is a string, this is interpreted as a filepath to load a previously saved model from a file, and all other arguments are overriden.
			saved = load(data_frame)
			for var in vars(saved).keys():
				setattr(self,var,vars(saved)[var])
	def predict(self,X=None):
		if X is not None:
			self.X_test = X[self.scalar_cols]
		self.y_pred = pd.Series(self.model.predict((self.X_test-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=self.X_test.index)
		try:
			delattr(self,'y_pred_proba')
		except:
			pass
		return self.y_pred
	def predict_proba(self,X=None):
		if X is not None:
			self.X_test = X[self.scalar_cols]
		self.y_pred_proba = pd.DataFrame(self.model.predict_proba((self.X_test-self.rescale_factors['minima'])/self.rescale_factors['ranges']),index=self.X_test.index,columns=['P(%s)'%s for s in self.class_labels])
		self.y_pred = self.y_pred_proba.idxmax(axis=1).map({'P(%s)'%s:s for s in self.class_labels}).rename('Predicted Class')
		return self.y_pred_proba
	def confusion_matrix(self,X=None,y_true=None,sample_weight=None):
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
		return self.cm
	def confusion_matrix_plot(self,X=None,y_true=None,savepath=None,sample_weight=None,title=None,figsize=(6.4,4.8),cmap='hot_r'):
		if title is None:
			title = 'Support Vector Machine Model Confusion Matrix'
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted')) #Setting index the same as columns, as I wanted, raises an error when feeding cm to sns.heatmap, I so had to hack it like so.
		fig, ax = plt.subplots(figsize=figsize)
		sns.heatmap(self.cm,cmap=cmap,vmax=max([self.cm[col].nlargest(2)[1] for col in self.cm.columns]),annot=True, linewidths=0.02, linecolor='k', fmt='d', ax=ax)
		ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
		ax.set_title(title)
		fig.tight_layout()
		if savepath is None:
			plt.show()
		else:
			fig.savefig(savepath)
	def confusion_metrics(self,X=None,y_true=None,sample_weight=None):
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					self.y_test, y_pred = X[y_true], self.predict(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					self.y_test, y_pred = y_true, self.predict(X)
			else:
				self.y_test, y_pred  = X[self.y_train.name], self.predict(X.drop(columns=self.y_train.name))
				y_true = self.y_test
			self.cm = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=self.class_labels,sample_weight=sample_weight),index=pd.Series(['%s'%s for s in self.class_labels],name='Actual'),columns=pd.Series(self.class_labels,name='Predicted'))
		acc = np.trace(self.cm)/self.cm.values.sum()
		err = 1- acc
		if len(self.class_labels)==2:
			p = self.pos_label
			n = [x for x in self.class_labels if x!= p][0]
			sens = self.cm.loc['%s'%p,p]/self.cm.loc['%s'%p,:].sum()
			spec = self.cm.loc['%s'%n,n]/self.cm.loc['%s'%n,:].sum()
			ppv = self.cm.loc['%s'%p,p]/self.cm.loc[:,p].sum()
			fpr = self.cm.loc['%s'%n,p]/self.cm.loc['%s'%n,:].sum()
			return {'Accuracy':acc,'Error Rate':err,'Sensitivity':sens,'Specificity':spec,'Positive Predictive Value':ppv,'False Positive Rate':fpr}
		return {'Accuracy':acc,'Error Rate':err}
	def roc_auc(self,X=None,y_true=None,plot=True,savepath=None,sample_weight=None,drop_intermediate=True,title=None,figsize=(6.4,4.8)):
		if title is None:
			title = 'Support Vector Machine Model ROC Curve'
		if len(self.class_labels)!=2:
			return None
		if X is not None:
			if y_true is not None:
				if type(y_true) == str:
					try:
						self.y_test, y_score = X[y_true], self.predict_proba(X.drop(columns=y_true))['P(%s)'%(self.pos_label)]
					except AttributeError:
						self.y_test, y_score = X[y_true], self.model.decision_function(X.drop(columns=y_true))
					y_true = self.y_test
				else:
					try:
						self.y_test, y_score = y_true, self.predict_proba(X)['P(%s)'%(self.pos_label)]
					except AttributeError:
						self.y_test, y_score = y_true, self.model.decision_function(X)
			else:
				try:
					self.y_test, y_score = X[self.y_train.name], self.predict_proba(X.drop(columns=self.y_train.name))['P(%s)'%(self.pos_label)]
				except AttributeError:
					self.y_test, y_score = X[self.y_train.name], self.model.decision_function(X.drop(columns=self.y_train.name))
				y_true = self.y_test
		else:
			try:
				y_true, y_score = self.y_test, self.predict_proba()['P(%s)'%(self.pos_label)]
			except AttributeError:
				y_true, y_score = self.y_test, self.model.decision_function(self.X_test)
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
		return area
	def save(self,savepath='svm_classifier.joblib'):
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