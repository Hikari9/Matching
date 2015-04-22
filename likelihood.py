
# to do: map caching with operations
#			(eg) dot product must cache row/column vectors appropriately
#	   : make caching toggle-able
#	   : apply pickling


# create likelihood (labelled pandas) matrix
# vectorizer can be 'tfidf' or 'count'

import numpy as np
import pandas as pd

# initializes a likelihood data frame based on row and column labels.
# Performs a content-based filtering technique using features.
# vectorizer can be 'tfidf' or 'count'
def likelihood_matrix(row_labels, column_labels, vectorizer='tfidf', ngram_range=(3,4)):

	rows = len(row_labels)
	columns = len(column_labels)
	
	# start vectorizing
	from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
	
	if vectorizer == 'tfidf': vectorizer = TfidfVectorizer
	elif vectorizer == 'count': vectorizer = CountVectorizer
	else: raise 'Vectorizer should be tfidf or count'
	
	vectorizer = vectorizer(analyzer='char', ngram_range=ngram_range)
	
	# extract vectors
	unique_labels = list(set(column_labels).union(set(row_labels)))
	features = vectorizer.fit_transform(unique_labels).toarray()
	
	# cache a mapping of label to corresponding vector
	vector_cache = {label: vector for label, vector in zip(unique_labels, features)}
	
	# get all pairwise cosine similarity
	# this will be the initial content of the matrix
	
	row_vectors = [vector_cache[label] for label in row_labels]
	column_vectors = [vector_cache[label] for label in column_labels]
	
	from sklearn.metrics.pairwise import cosine_similarity
	
	matrix = cosine_similarity(row_vectors, column_vectors)
	
	
	# create pandas data frame object
	data_frame = pd.DataFrame(matrix, index=row_labels, columns=column_labels)
	
	
	# append attributes
	
	# data_frame.has_cache = True
	data_frame.unique_labels = unique_labels # recalculate on dot
	data_frame.row_vectors = row_vectors # recalculate on dot
	data_frame.column_vectors = column_vectors # recalculate on dot
	data_frame.features = features # recalculate on dot
	data_frame.vector_cache = vector_cache # recalculate on dot
	data_frame.vectorizer = vectorizer # recalculate on dot
	data_frame.row_similarity_cache = {} # keep on dot
	data_frame.column_similarity_cache = {} # swap on dots
	
	# normalize data frame
	
	return data_frame

# clones attributes from first dataframe to next
def copy_dataframe_cache(origin_data_frame, destination_data_frame):
	self = origin_data_frame
	other = destination_data_frame
	
	if self.shape != other.shape:
		raise 'Dimensions should be the same!\nExpected: ' + self.shape + '\nFound: ' + other.shape
	
	# other.has_cache = self.has_cache
	other.unique_labels = self.unique_labels
	other.row_vectors = self.row_vectors
	other.column_vectors = self.column_vectors
	other.features = self.features
	other.vector_cache = self.vector_cache
	other.vectorizer = self.vectorizer
	other.row_similarity_cache = self.row_similarity_cache
	other.column_similarity_cache = self.column_similarity_cache



# extract features of a text in a form of a
# vector with respect to the labels in the data frame
def extract_features(data_frame, text):
	df = data_frame
	if text in df.vector_cache: 
		vector = df.vector_cache[text]
	else:
		vector = df.vectorizer.transform([text]).toarray()[0]
		df.vector_cache[text] = vector
	return vector




# returns the vector of similarity such that
# a certain text has features similar to corresponding
# set of labels
def similarity_vector(data_frame, text, use_row_labels = True):
	
	from sklearn.metrics.pairwise import cosine_similarity as cosines
	
	df = data_frame
	cache = df.row_similarity_cache if use_row_labels else df.column_similarity_cache
	
	#check first if exists in cache for performance
	if text in cache:
		similarities = cache[text]
	else:
		text_features = extract_features(df, text)
		
		if use_row_labels: # downward vector
			X, Y = df.row_vectors, [text_features]
			
		else: # righward vector
			X, Y = [text_features], df.column_vectors
			
		similarities = cosines(X, Y)
		cache[text] = similarities
	
	return similarities
	

# computes delta likelihood using the following steps: (non-verbatim)
# clabels = df.column_labels
# rlabels = df.row_labels
# vr = [cos_similarity(ur, label) for label in rlabels]
# vc = [cos_similarity(vr, label) for label in clabels]
# dL = vr * vc^T
def delta_likelihood(data_frame, user_row, user_column):
	
	# get similarity vectors
	# note: row similarities is downward while column_similarities is righward
	row_similarities = similarity_vector(data_frame, user_row, use_row_labels=True)
	column_similarities = similarity_vector(data_frame, user_column, use_row_labels=False)
	
	# combine computations by multiplying
	matrix = np.matrix(row_similarities) * np.matrix(column_similarities)
	delta = pd.DataFrame(matrix, index=df.index, columns=df.columns)
	
	# make sure both data frames has the same cache
	copy_dataframe_cache(data_frame, delta)
	
	return delta
	
	
