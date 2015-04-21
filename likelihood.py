# create likelihood (labelled pandas) matrix
# vectorizer can be 'tfidf' or 'count'

import numpy as np
import pandas as pd

# initializes a likelihood data frame based on row and column labels.
# Performs a content-based filtering technique using features.
# vectorizer can be 'tfidf' or 'count'
def likelihood_matrix(row_labels, column_labels, vectorizer='tfidf', ngram_range=(2,3)):

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
	
	data_frame.row_labels = row_labels
	data_frame.column_labels = column_labels
	data_frame.unique_labels = unique_labels
	data_frame.row_vectors = row_vectors
	data_frame.column_vectors = column_vectors
	data_frame.features = features
	data_frame.vector_cache = vector_cache
	data_frame.vectorizer = vectorizer
	data_frame.row_similarity_cache = {}
	data_frame.column_similarity_cache = {}
	
	return data_frame


# computes delta likelihood using the following steps: (non-verbatim)
# clabels = df.column_labels
# rlabels = df.row_labels
# vr = [cos_similarity(ur, label) for label in rlabels]
# vc = [cos_similarity(vr, label) for label in clabels]
# dL = vr * vc^T

def delta_likelihood(data_frame, user_row, user_column):
	
	df = data_frame
	row_vectors = df.row_vectors
	column_vectors = df.column_vectors
	
	def get_vector(text):
		if text in df.vector_cache: 
			vector = df.vector_cache[text]
		else:
			vector = df.vectorizer.transform([text]).toarray()[0]
			df.vector_cache[text] = vector
		return vector
		
	# get cosine similarities from user data to corresponding labels
	from sklearn.metrics.pairwise import cosine_similarity
	
	# check first if exists in cache for performance
	if user_row in df.row_similarity_cache:
		row_similarities = df.row_similarity_cache[user_row]
	else:
		user_row_vector = get_vector(user_row)
		# untransposed vector matrix
		row_similarities = cosine_similarity(row_vectors, [user_row_vector])
		# add back to cache
		df.row_similarity_cache[user_row] = row_similarities
	
	# do the same for columns
	if user_column in df.column_similarity_cache:
		column_similarities = df.column_similarity_cache[user_column]
	else:
		user_column_vector = get_vector(user_column)
		# transposed vector matrix
		column_similarities = cosine_similarity([user_column_vector], column_vectors)
		# add back to cache
		df.column_similarity_cache[user_column] = column_similarities
	
	
	# combine computations by multiplying
	matrix = np.matrix(row_similarities) * np.matrix(column_similarities)
	return pd.DataFrame(matrix, index=df.row_labels, columns=df.column_labels)
	
	