# create likelihood (labelled pandas) matrix
# vectorizer can be 'tfidf' or 'count'

import numpy as np
import pandas as pd

# initializes a zero matrix data frame based on row and column labels.
# vectorizer can be 'tfidf' or 'count'
def likelihood_matrix(row_labels, column_labels, vectorizer='tfidf', ngram_range=(2,4)):

	rows = len(row_labels)
	columns = len(column_labels)
	matrix = np.zeros((rows, columns))
	
	data_frame = pd.DataFrame(matrix, index=row_labels, columns=column_labels)
	
	data_frame.row_labels = row_labels
	data_frame.column_labels = column_labels
	
	# add a 'features' attribute to data frame
	from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
	
	if vectorizer == 'tfidf':
		vectorizer = TfidfVectorizer
	elif vectorizer == 'count':
		vectorizer = CountVectorizer
	else:
		return data_frame # don't add features
	
	vectorizer = vectorizer(analyzer='char', ngram_range=ngram_range)
	
	df = data_frame
	df.unique_labels = list(set(df).union(set(df.index)))
	df.features = vectorizer.fit_transform(df.unique_labels).toarray()
	df.cache = {label: vector for label, vector in zip(df.unique_labels, df.features)}
	df.vectorizer = vectorizer
	
	return df
	



# returns similarity function for the cosine similarity
# between vectorized labels in a data frame
def default_similarity(data_frame):

	df = data_frame
	
	def get_vector(text):
		if text not in df.cache:
			df.cache[text] = df.vectorizer.transform(text).toarray()
		return df.cache[text]
		
	
	from sklearn.metrics.pairwise import cosine_similarity
	
	def calculate_similarity(text1, text2):
		v1 = get_vector(text1)
		v2 = get_vector(text2)
		return cosine_similarity(v1, v2)[0][0]
	
	return calculate_similarity


# computes delta likelihood using the following function:
#	clabels = df
#	rlabels = df.index
#	vr = [similarity(ur, label) for label in rlabels]
#	vc = [similarity(vr, label) for label in clabels]
# 	dL = vr * vc^T
def delta_likelihood(data_frame, user_row, user_column, similarity = None):
	
	if similarity == None:
		similarity = default_similarity(data_frame)
	
	df = data_frame
	row_labels = df.row_labels
	column_labels = df.column_labels
	
	ur = user_row
	uc = user_column
	
	vr = [similarity(ur, label) for label in row_labels]
	vc = [similarity(uc, label) for label in column_labels]
	
	vr = np.matrix([[x] for x in vr]) # transposed
	vcT = np.matrix(vc) # untransposed
	
	return pd.DataFrame((vr * vcT).A, index=row_labels, columns=column_labels)
	
	
