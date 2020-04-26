'''
Library to hold my versions of common functions and classes for machine learning analysis of bioinformatic data.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from math import log10, exp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier



def categorical_palette(categories,uncat='-1',h=0,s=1,l=0.33): #auxiliary function to build dictionary of color codes assigned to categories on, e.g., a heatmap.  For n_categories<=12, pick from in-built categorical color palette.  For larger numbers, generate series of hues evenly spaced radially in HSL color space--h,l,s parameters only relevant in this case, to pass to sns.hls_palette.  The uncat parameter is used to identify label indicating uncategorized data points; this category will be assigned a grayscale value (and the remaining number of categories will be used to determine cutoff between using in-built vs generated palette.)
	paired_palette = sns.color_palette('Paired') #inbuilt matplotlib catergorical palette
	paired_palette = [paired_palette[i] for i in [3,1,9,11,10,4,0,7,5,8,2,6]] #re-order colors to taste
	if uncat in categories:
		categories = [c for c in categories if c != uncat] #uncategorized samples, if they exist, to be assigned black, and removed from consideration before generating the category palette
		length, m = len(categories), max((3*l-1)/2,0)
		if length > 12:
			cmap = sns.hls_palette(length,h=h,s=s,l=l)
			categories, cmap = [uncat]+categories,  [(m,m,m)]+[cmap[i//2 + ((length+1)//2)*(i%2)] for i in range(length)] #the palette was originally generated as a series of colors evenly spaced through the spectrum.  To maintain sharp boundaries even with many categories, when the transitions might otherwise become too subtle in e.g. a sorted heatmap, this step rearranges the colors' order by alternately picking from the front and back halves of the spectrum, maximizing the contrast of adjacent colors
		else:
			categories, cmap = [uncat]+categories, [(m,m,m)]+sns.color_palette(paired_palette[:length])
	else:
		length = len(categories)
		if length > 12:
			cmap = sns.hls_palette(length,h=h,s=s,l=l)
			cmap = [cmap[i//2 + ((length+1)//2)*(i%2)] for i in range(length)] 
		else:
			cmap = sns.color_palette(paired_palette[:length])
	return dict(zip(categories,cmap))



def no_float(x): #for use in generating category legends; need to both have numerical categories coerced to integers for display, and consistently make sure category labels are strings for some of the necessary calculations
	try:
		return str(int(x))
	except ValueError:
		return str(x)



def label_sort(x): #default key to sort numeric labels properly for legend display by adding leading zeros to numbers
	try:
		return '{:0>3d}'.format(int(x))
	except ValueError:
		return x



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



def roc_auc(model,x=None,savepath=None,sample_weight=None,drop_intermediate=True,title='ROC Curve'): #to be called as method on binary classifiers
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
	if savepath is None:
		plt.show()
		plt.close()
	else:
		plt.savefig(savepath)
		plt.close()
	return area




def heatmap(df,categ_col=None,categ_row=None,categ_col_uncat='-1',categ_row_uncat='-1',row_sort=True,col_sort=True,savepath=None,figsize=(6.4,4.8),title=None,cmap=None,center=None,vmin=None,vmax=None,method='average',metric='euclidean',z_score=None,row_cluster=False,col_cluster=False,row_linkage=None,col_linkage=None,return_reordered=False): #categ_col and categ_row parameters used to determine display of a column of colors for row categories or a row of colors for column categories respectively; categ_col_uncat and categ_row_uncat are the respective labels indicating unknown category; they will be assigned a black color label; savepath used to save figure instead of plt.show() default; title parameter used to add title to figure. row_sort and col_sort used to determine orderings for display and only apply as long as row_cluster/col_cluster are False; default is to sort according to category label ascending, numerical labels followed by string, but setting row_sort/col_sort to False or inputing no category column/row will leave order unchanged and inputting a dictionary, string function, or iterable will impose an ordering. (Even if row_cluster/col_cluster True, row_sort/col_sort can be used to order categories in legend). All other parameters passed to seaborn.clustermap
	#Set colormap for main heatmap, if necessary:
	data_frame = df.copy()
	if cmap is None:
		if center is None:
			cmap = 'mako'
		else:
			cmap = sns.diverging_palette(133,240,s=100,as_cmap=True)
	#Calculate row sorting order
	if categ_col is not None:
		if type (categ_col) != str: #assumption is that categ_col is part of data_frame, so need to add it if it's been provided as a separate iterable
			data_frame['categ_col'] = pd.Series(categ_col,index=data_frame.index)
			categ_col = 'categ_col'
		data_frame[categ_col] = data_frame[categ_col].apply(no_float)
		if type(row_sort)==bool:
			row_sorted = sorted(data_frame.index,key=lambda x: label_sort(data_frame.loc[x,categ_col]))
		elif callable(row_sort):
			row_sorted = sorted(data_frame.index,key=lambda x: row_sort(data_frame.loc[x,categ_col]))
		else:
			if type(row_sort)!=dict:
				row_sort = dict(zip([no_float(x) for x in row_sort],range(len(row_sort))))
			else:
				row_sort = dict(zip([no_float(x) for x in row_sort.keys()],row_sort.values()))
			row_sorted = sorted(data_frame.index,key=lambda x: row_sort[data_frame.loc[x,categ_col]])
	#Calculate col sorting order
	if categ_row is not None:
		if type(categ_row) != str: #assumption is that categ_row is part of data_frame, so need to add it if it's been provided as a separate iterable
			data_frame.loc['categ_row',:] = pd.Series(categ_row,index=data_frame.columns)
			categ_row = 'categ_row'
		data_frame.loc[categ_row] = data_frame.loc[categ_row].apply(no_float)
		if type(col_sort)==bool:
			col_sorted = sorted(data_frame,key=lambda x: label_sort(data_frame.loc[categ_row,x]))
		elif callable(col_sort):
			col_sorted = sorted(data_frame,key=lambda x: col_sort(data_frame.loc[categ_row,x]))
		else:
			if type(col_sort)!=dict:
				col_sort = dict(zip([no_float(x) for x in col_sort],range(len(col_sort))))
			else:
				col_sort = dict(zip([no_float(x) for x in col_sort.keys()],col_sort.values()))				
			col_sorted = sorted(data_frame,key=lambda x: col_sort[data_frame.loc[categ_row,x]])
	#generate heatmap, including generating colormap for categories among rows and/or columns:
	if categ_col is None and categ_row is None:
		g = sns.clustermap(data_frame,figsize=figsize,cmap=cmap,center=center,vmin=vmin,vmax=vmax,method=method,metric=metric,z_score=z_score,row_cluster=row_cluster,col_cluster=col_cluster,row_linkage=row_linkage,col_linkage=col_linkage)
	elif categ_row is None: #represent categories as a column of coded colors, independent of the main heatmap's colormap
		counts = dict(data_frame[categ_col].value_counts())
		if row_sort==False:
			categories = counts.keys()
		elif row_sort==True:
			categories = sorted(counts.keys(),key=label_sort)
		elif callable(row_sort):
			categories = sorted(counts.keys(),key=row_sort)
		else:
			categories = sorted(counts.keys(),key=lambda x: row_sort[x])
		catmap = categorical_palette(categories,categ_col_uncat)
		handles = [Patch(color=catmap[category],label=category+' (%i)'%counts[category]) for category in categories]
		if not row_cluster and row_sort!=False:
			data_frame = data_frame.loc[row_sorted,:]
		g = sns.clustermap(data_frame.drop(columns=categ_col),figsize=figsize,cmap=cmap,center=center,vmin=vmin,vmax=vmax,method=method,metric=metric,z_score=z_score,row_cluster=row_cluster,col_cluster=col_cluster,row_linkage=row_linkage,col_linkage=col_linkage,row_colors=data_frame[categ_col].rename('').map(catmap))
		ncol = 1 + len(handles)//33
	elif categ_col is None: #represent categories as a row of coded colors, independent of the main heatmap's colormap
		counts = dict(data_frame.loc[categ_row].value_counts())
		if col_sort==False:
			categories = counts.keys()
		elif col_sort==True:
			categories = sorted(counts.keys(),key=label_sort)
		elif callable(col_sort):
			categories = sorted(counts.keys(),key=col_sort)
		else:
			categories = sorted(counts.keys(),key=lambda x: col_sort[x])
		catmap = categorical_palette(categories,categ_row_uncat)
		handles = [Patch(color=catmap[category],label=category+' (%i)'%counts[category]) for category in categories]
		if not col_cluster and col_sort!=False:
			data_frame = data_frame[col_sorted]
		g = sns.clustermap(data_frame.drop(index=categ_row).astype('float64'),figsize=figsize,cmap=cmap,center=center,vmin=vmin,vmax=vmax,method=method,metric=metric,z_score=z_score,row_cluster=row_cluster,col_cluster=col_cluster,row_linkage=row_linkage,col_linkage=col_linkage,col_colors=data_frame.loc[categ_row].rename('').map(catmap)) #need to coerce data_frame dtypes back to float, in case row of string categories forced object dtype for each column
		ncol = 1 + len(handles)//33
	else: #if both a category row and category column
		counts_col = dict(data_frame.drop(index=categ_row)[categ_col].value_counts())
		if row_sort==False:
			categories_col = counts_col.keys()
		elif row_sort==True:
			categories_col = sorted(counts_col.keys(),key=label_sort)
		elif callable(row_sort):
			categories_col = sorted(counts_col.keys(),key=row_sort)
		else:
			categories_col = sorted(counts_col.keys(),key=lambda x: row_sort[x])
		counts_row = dict(data_frame.drop(columns=categ_col).loc[categ_row].value_counts())
		if col_sort==False:
			categories_row = counts_row.keys()
		elif col_sort==True:
			categories_row = sorted(counts_row.keys(),key=label_sort)
		elif callable(col_sort):
			categories_row = sorted(counts_row.keys(),key=col_sort)
		else:
			categories_row = sorted(counts_row.keys(),key=lambda x: col_sort[x])
		catmap_col = categorical_palette(categories_col,categ_col_uncat)
		catmap_row = categorical_palette(categories_row,categ_row_uncat)
		handles_row = [Patch(color=catmap_row[category],label=category+' (%i)'%counts_row[category]) for category in categories_row]
		handles_col = [Patch(color=catmap_col[category],label=category+' (%i)'%counts_col[category]) for category in categories_col]
		if not col_cluster and col_sort!=False:
			if categ_col not in col_sorted:
				col_sorted.append(categ_col)
			data_frame = data_frame[col_sorted]
		if not row_cluster and row_sort!=False:
			if categ_row not in row_sorted:
				row_sorted.append(categ_row)
			data_frame = data_frame.loc[row_sorted,:]
		g = sns.clustermap(data_frame.drop(columns=categ_col).drop(index=categ_row).astype('float64'),figsize=figsize,cmap=cmap,center=center,vmin=vmin,vmax=vmax,method=method,metric=metric,z_score=z_score,row_cluster=row_cluster,col_cluster=col_cluster,row_linkage=row_linkage,col_linkage=col_linkage,row_colors=data_frame[categ_col].rename('').map(catmap_col),col_colors=data_frame.loc[categ_row].rename('').map(catmap_row)) #need to coerce data_frame dtypes back to float, in case row of string categories forced object dtype for each column
		legend_col_ncol, legend_row_ncol = 1 + len(handles_col)//33, 1 + len(handles_row)//33
	g.cax.remove() #get rid of default colorbar
	fig, ax = g.fig, g.ax_heatmap
	if title is not None:
		title_text = fig.suptitle(title)
	#Create legend:
	if categ_col is not None and categ_row is not None:
		if data_frame.index.name is None:
			legend_title = 'vertical\ncategories'
		else:
			legend_title = data_frame.index.name+'\ncategories'
		legend_col = fig.legend(handles=handles_col,title=legend_title,bbox_to_anchor=(0.99,0.5),loc='center right',bbox_transform=fig.transFigure,borderaxespad=0.,fontsize=6,title_fontsize=6,ncol=legend_col_ncol)
		legend_col_width = legend_col.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure).width
		legend_col_height = legend_col.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure).height
		if data_frame.columns.name is None:
			legend_title = 'horizontal\ncategories'
		else:
			legend_title = data_frame.columns.name+'\ncategories'
		legend_row = fig.legend(handles=handles_row,title=legend_title,bbox_to_anchor=(0.98-legend_col_width,0.5),loc='center right',bbox_transform=fig.transFigure,borderaxespad=0.,fontsize=6,title_fontsize=6,ncol=legend_row_ncol)
		legend_row_width = legend_row.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure).width
		legend_row_height = legend_row.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure).height
		cb_rightshift = legend_row_width+0.01+legend_col_width+0.03 #make room for legends by shortening colorbar
		total_height = legend_col_height + legend_row_height + 0.01 #height of legends plus padding if they were to be vertically stacked
		if total_height < 1: #stack legends if there's room
			legend_col._loc_real  = 1 #upper right corner for loc
			legend_row._loc_real = 4 #lower right corner for loc
			legend_col.set_bbox_to_anchor((1,(1+total_height)/2),transform=fig.transFigure)
			legend_row.set_bbox_to_anchor((1,(1-total_height)/2),transform=fig.transFigure)
			cb_rightshift = max(legend_col_width,legend_row_width)+0.03 #make room for legend by shortening colorbar
	elif categ_col is not None or categ_row is not None:
		if categ_col is None:
			if data_frame.columns.name is None:
				legend_title = 'categories'
			else:
				legend_title = data_frame.columns.name+'\ncategories'
		else:
			if data_frame.index.name is None:
				legend_title = 'categories'
			else:
				legend_title = data_frame.index.name+'\ncategories'
		legend = fig.legend(handles=handles,title=legend_title,bbox_to_anchor=(0.99,0.5),loc='center right',bbox_transform=fig.transFigure,borderaxespad=0.,fontsize=6,title_fontsize=6,ncol=ncol)
		legend_width = legend.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure).width
		cb_rightshift = legend_width + 0.03 #make room for legend by shortening colorbar
	else:
		cb_rightshift = 0.03
	#Create colorbar and make other modificaitions to figure:
	cb_ax_loc = [0.02,0.06,0.98-cb_rightshift,0.04] #in figure coordinates, left margin, bottom margin, width, and height for new colorbar axis
	cb_ax = fig.add_axes(cb_ax_loc) #create new axes solely for colorbar
	mappable = ax.collections[0] #extract information on mapping of data points to colors for generating new colorbar
	c = fig.colorbar(mappable,cax=cb_ax,orientation='horizontal') #generate colorbar
	c.ax.tick_params(labelsize=6)
	ax.set_xlabel(ax.get_xlabel(),fontsize=6)
	ax.set_ylabel(ax.get_ylabel(),fontsize=6)
	if 	len(data_frame.columns) > 30:
		ax.set_xticks([]) #suppress ticks and labels in case there are too many to be helpful
	else:
		ax.set_xticklabels(ax.get_xmajorticklabels(),fontsize=6)
	if len(data_frame.index) > 20:
		ax.set_yticks([]) #suppress ticks and labels in case there are too many to be helpful
	else:
		ax.set_yticklabels(ax.get_ymajorticklabels(),fontsize=6)
	#Get default parameters for positioning of axes, associated text, and legends:
	cb_ax_tightbbox = c.ax.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	ax_tightbbox = ax.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	cb_ax_extent = c.ax.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	ax_extent = ax.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	row_dendrogram_extent = g.ax_row_dendrogram.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	col_dendrogram_extent = g.ax_col_dendrogram.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	if categ_col is not None:
		row_colors_extent = g.ax_row_colors.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	if categ_row is not None:
		col_colors_extent = g.ax_col_colors.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	#Calculate parameters for repositioning subplots (heatmap, dendrograms, row/column colors):
	subplots_right = fig.subplotpars.right + cb_ax_tightbbox.x1 - ax_tightbbox.x1 #align right edge of heatmap tightbbox to colorbar tightbbox
	if row_cluster: #align left edge of row dendrogram to left edge of colorbar
		subplots_left = fig.subplotpars.left - (row_dendrogram_extent.x0 - cb_ax_extent.x0)
	elif categ_col is not None:
		k = (fig.subplotpars.right-fig.subplotpars.left)/(ax_extent.x1-row_colors_extent.x0) #approximately constant
		extra_bbox_right = ax_tightbbox.x1 - ax_extent.x1 #constant
		subplots_left = subplots_right  - k * (cb_ax_tightbbox.x1 - extra_bbox_right - cb_ax_extent.x0)
	else:
		k = (fig.subplotpars.right-fig.subplotpars.left)/(ax_extent.x1-ax_extent.x0) #constant
		extra_bbox_right = ax_tightbbox.x1 - ax_extent.x1 #constant
		subplots_left = subplots_right  - k * (cb_ax_tightbbox.x1 - extra_bbox_right - cb_ax_extent.x0)
	subplots_bottom = fig.subplotpars.bottom + cb_ax_tightbbox.y1 + 0.03 - ax_tightbbox.y0 #align bottom edge of heatmap tightbbox 0.03 units above top edge of colorbar tightbbox
	if title is not None:
		top_edge = title_text.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure).y0 - 0.03
	else:
		top_edge = 0.97
	if col_cluster:	
		subplots_top = fig.subplotpars.top + (top_edge - col_dendrogram_extent.y1) #set top edge of col dendrogram
	elif categ_row is not None:
		k = (fig.subplotpars.top-fig.subplotpars.bottom)/(col_colors_extent.y1-ax_extent.y0)
		extra_bbox_bottom = ax_extent.y0 - ax_tightbbox.y0  #constant
		subplots_top = subplots_bottom + k * (top_edge - extra_bbox_bottom - cb_ax_tightbbox.y1 - 0.03)
	else:
		k = (fig.subplotpars.top-fig.subplotpars.bottom)/(ax_extent.y1-ax_extent.y0)
		extra_bbox_bottom = ax_extent.y0 - ax_tightbbox.y0  #constant
		subplots_top = subplots_bottom + k * (top_edge - extra_bbox_bottom - cb_ax_tightbbox.y1 - 0.03)
	#resize subplots within figure to make room for legend, colorbar, and text labels:
	subplots_loc = {'left':subplots_left,'right':subplots_right,'bottom':subplots_bottom,'top':subplots_top}
	fig.subplots_adjust(**subplots_loc)
	#Finally, show or save figure:
	if savepath is None:
		plt.show(fig)
	else:
		fig.savefig(savepath) #must use this method; g.savefig won't preserve adjustmnents made to layout
	plt.close(fig)
	if return_reordered:
		if row_cluster:
			row_ind = g.dendrogram_row.reordered_ind
		else:
			row_ind = list(range(len(data_frame.index)))
			if categ_row is not None:
				row_ind = row_ind[:len(row_ind)-1]
		if col_cluster:
			col_ind = g.dendrogram_col.reordered_ind
		else:
			col_ind = list(range(len(data_frame.columns)))
			if categ_col is not None:
				col_ind = col_ind[:len(col_ind)-1]
		return data_frame.iloc[row_ind,col_ind]

class pca:
	def __init__(self,data_frame,categ_col=None,n_components=None): #data_frame rows are samples, columns are variable names; assume data already recentered/scaled as appropriate
		if categ_col is None or type(categ_col) != str:
			self.data = data_frame.copy()
		else:
			self.data = data_frame.copy().drop(columns=categ_col)			
		pca = PCA(n_components=n_components,copy=False)
		if n_components is None:
			self.n_components = len(self.data.columns)
		else:
			self.n_components = n_components
		self.reduced = pd.DataFrame(pca.fit_transform(self.data),index=self.data.index,columns=['PC%i'%i for i in range(self.n_components)])
		self.components = pd.DataFrame(pca.components_,index=['PC%i'%i for i in range(self.n_components)],columns=self.data.columns)
		if categ_col is not None and type(categ_col) == str:
			self.reduced[categ_col] = data_frame[categ_col]
	def plot(self,savepath=None,categ_col=None,categ_sort=None,highlights=None,figsize=(6.4,4.8),title=None,uncat='-1'):
		with sns.axes_style('darkgrid'):
			fig, ax = plt.subplots(figsize=figsize)
			if title is not None:
				ax.set_title(title)
		if categ_col is None:
			self.reduced.plot.scatter(x='PC0',y='PC1',s=4,ax=ax,color='k')
		else:
			if type(categ_col) == str:
				categ_col = self.reduced[categ_col]
			else:
				categ_col = pd.Series(categ_col,index=self.reduced.index,dtype=str)
			categ_col = categ_col.apply(no_float)
			counts = dict(categ_col.value_counts())
			if categ_sort is None:
				categories = sorted(counts.keys(),key=label_sort)
			elif callable(categ_sort):
				categories = sorted(counts.keys(),key=categ_sort)
			else:
				if type(categ_sort)!=dict:
					categ_sort = dict(zip([no_float(x) for x in categ_sort],range(len(categ_sort))))
				else:
					categ_sort = dict(zip([no_float(x) for x in categ_sort.keys()],categ_sort.values()))
				categories = sorted(counts.keys(),key=lambda x: categ_sort[x])
			catmap = categorical_palette(categories,uncat)
			if highlights is None:
				handles = [Patch(color=catmap[category],label=category+' (%i)'%counts[category]) for category in categories]
			else:
				highlights = [no_float(x) for x in highlights]
				handles = [Patch(color=catmap[category],label=category+' (%i)'%counts[category]) for category in categories if category in highlights]
			if highlights is None:
				for category in categories:
					df = self.reduced[categ_col == category]
					df.plot.scatter(x='PC0',y='PC1',s=4,color=catmap[category],ax=ax)
			else:
				catmap2 = categorical_palette(categories,uncat,s=0.15,l=0.8)
				for category in categories:
					df = self.reduced[categ_col == category]
					if category in highlights:
						df.plot.scatter(x='PC0',y='PC1',s=4,color=catmap[category],ax=ax)
					else:
						df.plot.scatter(x='PC0',y='PC1',s=4,color=catmap2[category],ax=ax)
			ncol = 1 + len(handles)//35
			fig.legend(handles=handles,bbox_to_anchor=(0.99,0.5),loc='center right',borderaxespad=0.,fontsize=6,ncol=ncol)
			fig.subplots_adjust(left=0.07,right=(1.01-0.11*ncol))
		if savepath is None:
			plt.show()
		else:
			plt.savefig(savepath)
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
			if categ_col is None or type(categ_col) != str:
				self.data = data_frame.copy()
			else:
				self.data = data_frame.copy().drop(columns=categ_col)
			self.perplexity, self.early_exaggeration, self.learning_rate, self.metric, self.init = perplexity, early_exaggeration, learning_rate, metric, init
			self.embedding = pd.DataFrame(TSNE(perplexity=perplexity,early_exaggeration=early_exaggeration,learning_rate=learning_rate,metric=metric,init=init,verbose=verbose).fit_transform(self.data),index=self.data.index,columns=['tSNE 1','tSNE 2'])
			if categ_col is not None and type(categ_col) == str:
				self.embedding[categ_col] = data_frame[categ_col]
	def to_csv(self,filepath): #since tsne nondeterministic, useful to be able to save embeddings that look particularly good for later use
		header = 'perplexity: '+str(self.perplexity)+', early_exaggeration: '+str(self.early_exaggeration)+', learning_rate: '+str(self.learning_rate)+', metric: '+str(self.metric)+', init: '+str(self.init)
		pd.DataFrame(self.embedding.values,index=self.embedding.index,columns=pd.MultiIndex.from_arrays([[header]+[' ' for i in range(len(self.embedding.columns)-1)],self.embedding.columns])).to_csv(filepath) #MultiIndex trick to effectively add a header row
	def plot(self,savepath=None,categ_col=None,categ_sort=None,highlights=None,figsize=(6.4,4.8),title=None,uncat='-1'):
		with sns.axes_style('darkgrid'):
			fig, ax = plt.subplots(figsize=figsize)
			if title is not None:
				ax.set_title(title)
		if categ_col is None:
			self.embedding.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,ax=ax,color='k')
		else:
			if type(categ_col) == str:
				categ_col = self.embedding[categ_col]
			else:
				categ_col = pd.Series(categ_col,index=self.embedding.index,dtype=str)
			categ_col = categ_col.apply(no_float)
			counts = dict(categ_col.value_counts())
			if categ_sort is None:
				categories = sorted(counts.keys(),key=label_sort)
			elif callable(categ_sort):
				categories = sorted(counts.keys(),key=categ_sort)
			else:
				if type(categ_sort)!=dict:
					categ_sort = dict(zip([no_float(x) for x in categ_sort],range(len(categ_sort))))
				else:
					categ_sort = dict(zip([no_float(x) for x in categ_sort.keys()],categ_sort.values()))
				categories = sorted(counts.keys(),key=lambda x: categ_sort[x])
			catmap = categorical_palette(categories,uncat)
			if highlights is None:
				handles = [Patch(color=catmap[category],label=category+' (%i)'%counts[category]) for category in categories]
			else:
				highlights = [no_float(x) for x in highlights]
				handles = [Patch(color=catmap[category],label=category+' (%i)'%counts[category]) for category in categories if category in highlights]
			if highlights is None:
				for category in categories:
					df = self.embedding[categ_col == category]
					df.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,color=catmap[category],ax=ax)
			else:
				catmap2 = categorical_palette(categories,uncat,s=0.15,l=0.8)
				for category in categories:
					df = self.embedding[categ_col == category]
					if category in highlights:
						df.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,color=catmap[category],ax=ax)
					else:
						df.plot.scatter(x='tSNE 1',y='tSNE 2',s=4,color=catmap2[category],ax=ax)
			ncol = 1 + len(handles)//35
			fig.legend(handles=handles,bbox_to_anchor=(0.99,0.5),loc='center right',borderaxespad=0.,fontsize=6,ncol=ncol)
			fig.subplots_adjust(left=0.07,right=(1.01-0.11*ncol))
		if savepath is None:
			plt.show()
		else:
			plt.savefig(savepath)
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



def sculpt_tree(Z,n,min_samples=3): #cut dendrogram to give n categories subject to constraint that each category must have at least min_samples members
	m, max_cats, true_n_hist = n, len(Z)/min_samples, []
	labels = pd.Series(fcluster(Z,m,criterion='maxclust'))
	counts = labels.value_counts().sort_index()
	cats, true_n = list(counts.index), sum(counts>=min_samples)
	while true_n < n:
		m+=1
		if m > max_cats: #this break condition implicitly makes the assumption that true_n will keep increasing, which seems reasonable.  However, it may stop increasing sooner than absolute maximum number of categories hit, so could use some optimizing here still
			print('break')
			break
		labels = pd.Series(fcluster(Z,m,criterion='maxclust'))
		counts = labels.value_counts().sort_index()
		cats, true_n = list(counts.index), sum(counts>=min_samples)
	true_counts, orphan_counts = counts[counts>=min_samples], counts[counts<min_samples]
	true_cats, orphans = list(true_counts.index), list(orphan_counts.index)
	remap = dict(zip(true_cats,range(len(true_cats))))
	remap.update(dict(zip(orphans,[int(-1) for i in range(len(orphans))])))
	return labels.map(remap)

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

