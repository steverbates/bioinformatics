import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

def categorical_palette(categories,uncat='-1',h=0,s=1,l=0.35): #auxiliary function to build dictionary of color codes assigned to categories on, e.g., a heatmap.  For n_categories<=12, pick from in-built categorical color palette.  For larger numbers, generate series of hues evenly spaced radially in HSL color space--h,l,s parameters only relevant in this case, to pass to sns.hls_palette.  The uncat parameter is used to identify label indicating uncategorized data points; this category will be assigned black (and the remaining number of categories will be used to determine cutoff between using in-built vs generated palette.)
	paired_palette = sns.color_palette('Paired') #inbuilt matplotlib catergorical palette
	paired_palette = [paired_palette[i] for i in [3,1,9,11,10,4,0,7,5,8,2,6]] #re-order colors to taste
	if uncat in categories:
		categories = [c for c in categories if c != uncat] #uncategorized samples, if they exist, to be assigned black, and removed from consideration before generating the category palette
		length = len(categories)
		if length > 12:
			cmap = sns.hls_palette(length,h=h,s=s,l=l)
			categories, cmap = [uncat]+categories,  [(0,0,0)]+[cmap[i//2 + ((length+1)//2)*(i%2)] for i in range(length)] #the palette was originally generated as a series of colors evenly spaced through the spectrum.  To maintain sharp boundaries even with many categories, when the transitions might otherwise become too subtle in e.g. a sorted heatmap, this step rearranges the colors' order by alternately picking from the front and back halves of the spectrum, maximizing the contrast of adjacent colors
		else:
			categories, cmap = [uncat]+categories, [(0,0,0)]+sns.color_palette(paired_palette[:length])
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



def label_sort(x): #used in conjunction with no_float, as key to sort numeric labels properly for legend display by adding leading zeros
	try:
		return '{:0>3d}'.format(int(x))
	except ValueError:
		return x



def heatmap(data_frame,categ_col=None,categ_row=None,categ_col_uncat='-1',categ_row_uncat='-1',savepath=None,figsize=(6.4,4.8),title=None,cmap=None,center=None,vmin=None,vmax=None,method='average',metric='euclidean',z_score=None,row_cluster=False,col_cluster=False,row_linkage=None,col_linkage=None): #categ_col and categ_row parameters used to determine display of a column of colors for row categories or a row of colors for column categories respectively; categ_col_uncat and categ_row_uncat are the respective labels indicating unknown category; they will be assigned a black color label; savepath used to save figure instead of plt.show() default; title parameter used to add title to figure; all other parameters passed to seaborn.clustermap
	#Set colormap for main heatmap, if necessary:
	if cmap is None:
		if center is None:
			cmap = 'mako'
		else:
			cmap = sns.diverging_palette(133,240,s=100,as_cmap=True)
	#generate heatmap, including generating colormap for categories among rows and/or columns:
	if categ_col is None and categ_row is None:
		g = sns.clustermap(data_frame,figsize=figsize,cmap=cmap,center=center,vmin=vmin,vmax=vmax,method=method,metric=metric,z_score=z_score,row_cluster=row_cluster,col_cluster=col_cluster,row_linkage=row_linkage,col_linkage=col_linkage)
	elif categ_row is None: #represent categories as a column of alternating colors, independent of the main heatmap's colormap, in row_cluster=False case sorting rows so that each category is contiguous
		if type (categ_col) != str: #assumption is that categ_col is part of data_frame, so need to add it if it's been provided as a separate iterable
			data_frame['categ_col'] = pd.Series(categ_col,index=data_frame.index)
			categ_col = 'categ_col'
		data_frame[categ_col] = data_frame[categ_col].apply(no_float)
		counts = dict(data_frame[categ_col].value_counts())
		categories = sorted(counts.keys(),key=label_sort)
		catmap = categorical_palette(categories,categ_col_uncat)
		handles = [Patch(color=catmap[category],label=category+' (%i)'%counts[category]) for category in categories]
		if not row_cluster:
			data_frame = data_frame.loc[sorted(data_frame.index,key=lambda x: label_sort(data_frame.loc[x,categ_col])),:]
		g = sns.clustermap(data_frame.drop(columns=categ_col),figsize=figsize,cmap=cmap,center=center,vmin=vmin,vmax=vmax,method=method,metric=metric,z_score=z_score,row_cluster=row_cluster,col_cluster=col_cluster,row_linkage=row_linkage,col_linkage=col_linkage,row_colors=data_frame[categ_col].rename('').map(catmap))
		ncol = 1 + len(handles)//35
	elif categ_col is None: #represent categories as a row of alternating colors, independent of the main heatmap's colormap, col_cluster=False case sorting columns so that each category is contiguous
		if type(categ_row) != str: #assumption is that categ_row is part of data_frame, so need to add it if it's been provided as a separate iterable
			data_frame.loc['categ_row',:] = pd.Series(categ_row,index=data_frame.columns)
			categ_row = 'categ_row'
		data_frame.loc[categ_row] = data_frame.loc[categ_row].apply(no_float)
		counts = dict(data_frame.loc[categ_row].value_counts())
		categories = sorted(counts.keys(),key=label_sort)
		catmap = categorical_palette(categories,categ_row_uncat)
		handles = [Patch(color=catmap[category],label=category+' (%i)'%counts[category]) for category in categories]
		if not col_cluster:
			data_frame = data_frame[sorted(data_frame,key=lambda x: label_sort(data_frame.loc[categ_row,x]))]
		g = sns.clustermap(data_frame.drop(index=categ_row).astype('float64'),figsize=figsize,cmap=cmap,center=center,vmin=vmin,vmax=vmax,method=method,metric=metric,z_score=z_score,row_cluster=row_cluster,col_cluster=col_cluster,row_linkage=row_linkage,col_linkage=col_linkage,col_colors=data_frame.loc[categ_row].rename('').map(catmap)) #need to coerce data_frame dtypes back to float, in case row of string categories forced object dtype for each column
		ncol = 1 + len(handles)//35
	else: #if both a category row and category column
		if type (categ_col) != str: #assumption is that categ_col is part of data_frame, so need to add it if it's been provided as a separate iterable
			data_frame['categ_col'] = pd.Series(categ_col,index=data_frame.index)
			categ_col = 'categ_col'
		if type(categ_row) != str: #assumption is that categ_row is part of data_frame, so need to add it if it's been provided as a separate iterable
			data_frame.loc['categ_row',data_frame.columns.difference([categ_col])] = pd.Series(categ_row,index=data_frame.columns.difference([categ_col]))
			categ_row = 'categ_row'
		data_frame.loc[categ_row] = data_frame.loc[categ_row].apply(no_float)
		data_frame[categ_col] = data_frame[categ_col].apply(no_float)
		counts_col = dict(data_frame.drop(index=categ_row)[categ_col].value_counts())
		categories_col = sorted(counts_col.keys(),key=label_sort)
		counts_row = dict(data_frame.drop(columns=categ_col).loc[categ_row].value_counts())
		categories_row = sorted(counts_row.keys(),key=label_sort)
		catmap_col = categorical_palette(categories_col,categ_col_uncat)
		catmap_row = categorical_palette(categories_row,categ_row_uncat)
		handles_row = [Patch(color=catmap_row[category],label=category+' (%i)'%counts_row[category]) for category in categories_row]
		handles_col = [Patch(color=catmap_col[category],label=category+' (%i)'%counts_col[category]) for category in categories_col]
		if not col_cluster:
			data_frame = data_frame[sorted(data_frame,key=lambda x: label_sort(data_frame.loc[categ_row,x]))]
		if not row_cluster:
			data_frame = data_frame.loc[sorted(data_frame.index,key=lambda x: label_sort(data_frame.loc[x,categ_col])),:]
		g = sns.clustermap(data_frame.drop(columns=categ_col).drop(index=categ_row).astype('float64'),figsize=figsize,cmap=cmap,center=center,vmin=vmin,vmax=vmax,method=method,metric=metric,z_score=z_score,row_cluster=row_cluster,col_cluster=col_cluster,row_linkage=row_linkage,col_linkage=col_linkage,row_colors=data_frame[categ_col].rename('').map(catmap_col),col_colors=data_frame.loc[categ_row].rename('').map(catmap_row)) #need to coerce data_frame dtypes back to float, in case row of string categories forced object dtype for each column
		legend_col_ncol, legend_row_ncol = 1 + len(handles_col)//35, 1 + len(handles_row)//35
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
	elif categ_col is not None or categ_row is not None: #title variable already used to set figure title so it can be recycled
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
	row_dendrogram_tightbbox = g.ax_row_dendrogram.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	col_dendrogram_tightbbox = g.ax_col_dendrogram.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	cb_ax_extent = c.ax.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	ax_extent = ax.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	row_dendrogram_extent = g.ax_row_dendrogram.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	col_dendrogram_extent = g.ax_col_dendrogram.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	if categ_col is not None:
		row_colors_tightbbox = g.ax_row_colors.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
		row_colors_extent = g.ax_row_colors.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	if categ_row is not None:
		col_colors_tightbbox = g.ax_col_colors.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
		col_colors_extent = g.ax_col_colors.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	if categ_col is not None and categ_row is not None:
		legend_col_tightbbox = legend_col.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
		legend_row_tightbbox = legend_row.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
		legend_col_extent = legend_col.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
		legend_row_extent = legend_row.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
	elif categ_col is not None or categ_row is not None:
		legend_tightbbox = legend.get_tightbbox(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
		legend_extent = legend.get_window_extent(fig.canvas.get_renderer()).inverse_transformed(fig.transFigure)
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