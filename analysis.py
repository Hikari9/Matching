# linear algebra
import numpy as np
import pandas as pd

# text vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# for nearest neighbor
from sklearn.neighbors import NearestNeighbors

# cosine similarity
from sklearn.metrics.pairwise import linear_kernel as cosine_similarity

# note that we can use linear kernel because
# vectors from text vectorizers are already normalized


def down(nparray):
	return nparray.reshape(nparray.shape[0], 1)

def right(nparray):
	return nparray.reshape(1, nparray.shape[0])

'''
Likelihood Matrix
	- matrix quantifying the likelihood of a match
		between row and column elements
	- performs an initial content-based filtering upon construction
	- applicable for collaborative updating
	- nature of being a matrix allows chain multiplication
'''

class LikelihoodMatrix(object):

	def __str__(self):
		return str(self.dataframe)
	
	def _repr_html_(self):
		return self.dataframe.to_html()
	
	'''
	Constructor
		- initializes a likelihood data frame based on row and column labels
		- performs a content-based filtering technique using features
		- vectorizer can be 'tfidf' or 'count' 
		- uses cosine similarity for label matching
	'''
	
	def __init__(self, rows, columns, vectorizer='tfidf', ngram_range=(3,4), all_zeroes=False):
	
		# start vectorizing
		if vectorizer == 'tfidf':		vectorizer = TfidfVectorizer
		elif vectorizer == 'count':		vectorizer = CountVectorizer
		else:							raise ValueError("vectorizer should be 'tfidf' or 'count'")
		
		vectorizer = vectorizer(analyzer='char', ngram_range=ngram_range)
		
		# extract vectors
		unique_labels = list(set(rows).union(set(columns)))
		features = vectorizer.fit_transform(unique_labels).toarray()
		
		# cache a mapping of label to corresponding vector
		vector_cache = {label:vector for label, vector in zip(unique_labels, features)}
		
		# get all pairwise cosine similarity
		# this will be the initial content of the matrix
		row_vectors = [vector_cache[label] for label in rows]
		column_vectors = [vector_cache[label] for label in columns]
		
		# create matrix
		if all_zeroes:
			matrix = np.zeroes(len(rows) * len(columns)).reshape(len(rows), len(columns))
		else:
			matrix = cosine_similarity(row_vectors, column_vectors)
		
		# create pandas data frame object
		self.dataframe = pd.DataFrame(matrix, index=rows, columns=columns)
		
		# append important attributes
		self.rows = rows
		self.columns = columns
		self.row_vectors = row_vectors
		self.column_vectors = column_vectors
		self.vectorizer = vectorizer
		self.vector_cache = vector_cache
		# self.row_similarity_cache = {}
		# self.column_similarity_cache = {}
	
	
	'''
	Find Matches
		- a vector of quantified matches on opposite axis
		- equivalent to LM * down(SV2) (on_rows)
		- equivalent to right(SV1) * LM (not on_rows)
		- one can return a dataframe for readability
	'''
	
	def find_matches(self, text, on_rows = False, with_labels = True, percentage = True):
		similarities = self.similarity_vector(text, on_rows=not on_rows, with_labels=False, percentage=False)

		if on_rows:
			matrix = np.dot(self.dataframe.values, down(similarities))
			matches = matrix.reshape(matrix.shape[0])

		else:
			matches = np.dot(right(similarities), self.dataframe.values)[0]

		if percentage and matches.any():
			matches *= 100. / matches.sum()
		
		if with_labels:
			return pd.Series(matches, index=self.rows if on_rows else self.columns)
		else:
			return matches

	
	# convenience methods for match finding
	
	def find_row_matches(self, text, with_labels = True, percentage = True):
		return self.find_matches(text, on_rows=True, with_labels=with_labels, percentage=percentage)
	
	def find_column_matches(self, text, with_labels = True, percentage = True):
		return self.find_matches(text, on_rows=False, with_labels=with_labels, percentage=percentage)
	
	
	'''
	Add Match
		- updates the matrix count by adding the delta
		- uses frequency-based collaborative filtering
	'''
	
	def add_match(self, row_text, column_text):
		self.dataframe += self.delta(row_text, column_text, with_labels=False)

	'''
	Subtract Match
		- updates the matrix count by subtracting the delta
		- uses frequency-based collaborative filtering
	'''

	def subtract_match(self, row_text, column_text):
		self.dataframe -= self.delta(row_text, column_text, with_labels=False)
	
	'''
	Recommendation Score
		- obtain a quantified score of two labels
		- equivalent to right(SV1) * LM * down(SV2)
	'''
	
	def recommendation_score(self, row_text, column_text):
		SV1 = self.similarity_vector(row_text, on_rows=True, with_labels=False, percentage=True)
		SV2 = self.similarity_vector(column_text, on_rows=False, with_labels=False, percentage=False)
		LM = self.dataframe.values
		return right(SV1).dot(LM).dot(SV2)[0]
		
	'''
	Features Vector
		- extract relative features vector of a text
		- vectorizes with respect to dataframe labels
	'''
	
	def features_vector(self, text):
	
		try: # obtain from cache
			return self.vector_cache[text]
		except AttributeError:
			self.vector_cache = {}
		except KeyError:
			pass
			
		# not in cache. calculate first
		
		vector = self.vectorizer.transform([text]).toarray()[0]
		self.vector_cache[text] = vector
		
		return vector
	
	'''
	Similarity Vector
		- extract vector of cosine similarity on same axis
		- one can return a series for readability
	'''
	
	def similarity_vector(self, text, on_rows = True, with_labels = True, percentage = False):
		
		# function to process parameter options
		def process(similarities):
			if percentage and similarities.any():
				similarities *= 100.0 / similarities.sum()
			if with_labels:
				return pd.Series(similarities, index=self.rows if on_rows else self.columns)
			return similarities
		
		try: # obtain from cache
			cache = self.row_similarity_cache if on_rows else self.column_similarity_cache
			return process(cache[text])
		
		except AttributeError:
			if on_rows: self.row_similarity_cache = {}
			else: self.column_similarity_cache = {}
		
		except KeyError:
			pass
		
		# not in cache. calculate first
		
		text_features = self.features_vector(text)
		
		cache = self.row_similarity_cache if on_rows else self.column_similarity_cache
		X = [text_features]
		Y = self.row_vectors if on_rows else self.column_vectors
		
		cos_sim = cosine_similarity(X, Y)[0]
		cache[text] = cos_sim

		return process(cos_sim)
		
	
	'''
	Delta Matrix
		- garners which row/column labels the input most likely is
		- returns a matrix of most likely label match
		- uses content-based matching of features
		- note: the content of self.dataframe does not affect 
			the values returned by the matrix
		- computes delta likelihood matrix by multiplying
			the similarity vectors of two samples from both axes
		- equivalent to down(SV1) * right(SV2)
	'''
	
	def delta(self, row_text, column_text, with_labels = True):
		
		# get similarity vectors
		# note: downward for rows, rightward for columns
		SV1 = self.similarity_vector(row_text, on_rows=True, with_labels=False, percentage=False)
		SV2 = self.similarity_vector(column_text, on_rows=False, with_labels=False, percentage=False)
		

		#combine computations by multiplying
		matrix = np.dot(down(SV1), right(SV2))
		
		if with_labels:
			return pd.DataFrame(matrix, index=self.rows, columns=self.columns)
		else:
			return matrix
	
	
	'''
	Combining of Matrices
		- to derive combined likelihood, matrices are multiplied
		- note that matrix multiplication rules apply
	'''
	
	def __mul__(self, other):
		if isinstance(other, LikelihoodMatrix):
			product = LikelihoodMatrix(self.rows, other.columns)
			product.dataframe += self.dataframe.dot(other.dataframe)
			return product
		else:
			raise TypeError('Can only with multiply likelihood_matrix')
	
	
	'''
	Copy
		- returns a clone with a copy of attribute pointers
		- copy with cache is optional
	'''
	
	def copy(self, copy_cache = False):
		cls = self.__class__
		other = cls.__new__(cls)
		
		# copy important attribute pointers
		other.dataframe = self.dataframe
		other.rows = self.rows
		other.columns = self.columns
		other.row_vectors = self.row_vectors
		other.column_vectors = self.column_vectors
		other.vectorizer = self.vectorizer
		if copy_cache:
			names = [
				'vector_cache',
				'row_similarity_cache',
				'column_similarity_cache'
			]
			for name in names:
				try:
					cache = getattr(self, name)
				except AttributeError:
					pass
				else:
					setattr(other, name, cache)
	
	# for copy module
	def __copy__(self):
		return self.clone()