import re

# SECTION 1: individual filters

# recursively replaces all occurences
# e.g. recursive_replace("& &", "&", "& & & & &")
# returns "&"
def recursive_replace(to_find, to_replace, text):
	buffer = text
	while to_find in buffer:
		buffer = re.sub(to_find, to_replace, buffer)
	return buffer


# removes special characters and normalizes separators in a string
def remove_special_characters(remove_ampersand):
	
	to_trim = ['&', '/', 'IN', 'OF']
	
	def transform(text):
		buffer = ' ' + text + ' ' # extend whitespace for generality
		buffer = buffer.replace("'", "") # remove apostrophe completely
		buffer = buffer.replace('&', ' & ') # extend ampersand
		buffer = buffer.replace(' AND ', ' & ') # convert AND to ampersand
		if remove_ampersand:
			buffer = buffer.replace('&', '')
		buffer = buffer.replace('/', ' / ') # extend separator
		buffer = re.sub(r'[^a-zA-Z0-9&/]', ' ', buffer) # retain alphanumeric with &/
		buffer = re.sub(r'\s\s+', ' ', buffer) # remove contiguous whitespace
		buffer = recursive_replace('/ &', '/', buffer) # reduce redundant slash and ampersand to slash
		buffer = recursive_replace('& /', '/', buffer) # same as above
		
		buffer = buffer.strip()
		
		# change delimeter for repeated trimming of elements
		while True:
			modified = False
			buffer = ' ' + buffer + ' '
			for word in to_trim:
				while buffer.startswith(' ' + word + ' '):
					buffer = buffer[len(word)+1:]
					modified = True
				while buffer.endswith(' ' + word + ' '):
					buffer = buffer[:-len(word)-1]
					modified = True
			if not modified: break
		
		return buffer.strip()
	
	return transform

# transforms input to uppercase string
def to_uppercase(text):
	return str(text).upper()

# removes '# ___ Y_R[S]'
# e.g. 4th YEAR CS -> CS
def remove_year(text):
	return re.sub(r'[0-9].*Y.*[\S]]*[RS|R]', '', text).strip()


# transform abbreviations based a text file
# format of strings in text file:
#
# LONG 1
# 	SHORT 1
# 	SHORT 2
# 	...
# 	SHORT n
#
# LONG 2
# 	SHORT 1
# 	SHORT 2
# 	...
# 	SHORT n
# ...
def transform_abbreviations(filename):
	from algoutils import read
	lines = map(str.strip, read(filename)) # read file first
	
	longline = '\n'.join(lines)
	groups = longline.split('\n\n')
	
	groups = map(lambda string: string.split('\n'), groups)
	groups = map(lambda array: (array[0], array[1:]), groups)
	
	# at this point, every group is a tuple of (LONG, [SHORTs])
	
	pairs = []
	
	# collect all abbreviations
	for long, shorts in groups:
		for abbreviation in shorts:
			pairs.append((abbreviation, long))
			# map space-separated abbreviations, too
			pairs.append((' '.join(abbreviation), long))
	
	# extend whitespace
	pairs = map(lambda (a, b): (' ' + a + ' ', ' ' + b + ' '), pairs)
	
	# transform method to return
	def transform(text):
		buffer = ' ' + text + ' '
		for short, long in pairs:
			buffer = buffer.replace(short, long)
		return buffer.strip()
	
	return transform


# remove prefix from data in text file
# all lines in text file are considered
def remove_prefixes(filename):
	from algoutils import read
	prefixes = read(filename) # read file first
	
	def transform(text):
		buffer = text.strip() + ' '
		for prefix in prefixes:
			while buffer.startswith(prefix):
				buffer = buffer[len(prefix):].strip() + ' '
		
		return buffer.strip()
	
	return transform
		

# convert separators into slashes from data in text file
def transform_separators(filename, delimiter = '/'):
	from algoutils import read
	separators = read(filename) # read file first
	
	def transform(text):
		# extend whitespace first
		buffer = ' ' + text + ' '
		for separator in separators:
			buffer = buffer.replace(' ' + separator.strip() + ' ', ' ' + delimiter + ' ')
		buffer = recursive_replace(delimiter + ' ' + delimiter, delimiter, buffer)
		
		return buffer.strip()
	
	return transform


# removes duplicate words/expressions
# note: words separated by slashes (/)
# will be independent from each other
def remove_duplicate_words(text):
	words = text.split(' ')
	word_buffer = []
	visited = set()
	
	for word in words:
		if word == '/':
			visited.clear() # clear dictionary on separator
		if word != '&' and word in visited:
			continue
		word_buffer.append(word)
		visited.add(word)
	
	# join back words
	buffer = ' '.join(word_buffer)
	buffer = recursive_replace('& &', '&', buffer)
	buffer = recursive_replace('/ /', '/', buffer)
	return buffer

# respells a given word based on a dictionary (a set or a list of words)
# words are listed in a text file

_cached_dictionaries = {}

def respeller(dictionary_file, length_at_least, edit_at_most, enable_cache = True):
	
	global _cached_dictionaries
	
	# cache dictionary for performance
	
	if enable_cache and dictionary_file in _cached_dictionaries:
		print 'Loaded cached dictionary: <', dictionary_file, '>'
		trie = _cached_dictionaries[dictionary_file]
	else:
		from suffix import Trie
		from algoutils import read
		trie = Trie(map(str.strip, read(dictionary_file)))
		_cached_dictionaries = trie
		
	
		
	# have a suffix trie map suggestions through edit distance
	# to do: save or pickle trie
	
	def suggest(word):
		if len(word) < length_at_least or word in trie:
			return word
		neighbors = trie.search(word, k = edit_at_most)
		if len(neighbors) == 0:
			return word
		return min(neighbors, key = lambda (word, dist): dist)[0]
	
	def transform(data):
		return ' '.join(map(suggest, data.split(' ')))
	
	return transform

try: # import if corpus exists
	from nltk.stem import WordNetLemmatizer

except: # download corpora if does not exist
	import nltk
	if not nltk.download('wordnet'):
		raise Exception('Error in downloading wordnet. \
						Please make sure you are connected to the network, \
						or try downloading manually.')
	from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# lemmatize each word for a given text
def lemmatize(text):
	splitted = text.split(' ')
	transformed = [str(lemmatizer.lemmatize(word.lower()).upper()) if len(word) > 3 else word for word in splitted]
	return ' '.join(transformed)

# filters to be used to clean courses
course_filters = [
	to_uppercase,
	remove_special_characters
		(remove_ampersand = True),
	remove_year,
	remove_prefixes
		('data/course_prefixes.txt'),
	lemmatize,
	transform_abbreviations
		('data/course_abbreviations.txt'),
	transform_separators
		('data/course_prefixes.txt', delimiter = '/'),
	remove_duplicate_words,
	remove_special_characters
		(remove_ampersand = True)
]

# filters to be used to clean industries
industry_filters = [
	to_uppercase,
	remove_special_characters
		(remove_ampersand = False),
	remove_year,
	lemmatize,
	transform_abbreviations
		('data/industry_abbreviations.txt'),
	remove_duplicate_words,
	remove_special_characters
		(remove_ampersand = False)
]

# filters to be used to clean job titles
job_title_filters = [
	to_uppercase,
	remove_special_characters
		(remove_ampersand = False),
	remove_year,
	lemmatize,
	transform_abbreviations
		('data/job_title_abbreviations.txt'),
	remove_duplicate_words,
	remove_special_characters
		(remove_ampersand = False)
]

# recursive clean method to be used
# styles: course, industry, job title
# respell if dictionary file is provided

_filter_map = {
	'course': course_filters,
	'industry': industry_filters,
	'job title': job_title_filters
}

_dictionary_map = {
	'course': 'data/course_dictionary.txt',
	'industry': 'data/industry_dictionary.txt',
	'job title': 'data/job_title_dictionary.txt'
}

D_TYPE_INDUSTRY = 'industry'
D_TYPE_EDUC = 'course'
D_TYPE_JOB_TITLE = 'job title'


def clean(list_of_data, style, spell_check = True, dictionary_file = None, length_at_least = 6, edit_at_most = 2, enable_cache = True):
	
	# main method to filter
	# style must be from the dictionaries above
	# or one can opt to use D_TYPE_* constants
	try:
		global _filter_map
		filters = _filter_map[style]
	except:
		raise ValueError('Style must be ' + str(_filter_map.keys()) + ' or callable')
	
	if spell_check:
		try:
			if dictionary_file == None:
				dictionary_file = _dictionary_map[style]
			respell = respeller(dictionary_file, length_at_least, edit_at_most, enable_cache)
		except:
			dictionary_file = None

		
	def clean_single(data):
		try:
			data = data.encode('ascii', 'ignore')
		except:
			data = ''
		buffer = data
		for filter_ in filters:
			buffer = filter_(buffer)
		
		# respell if dictionary file is provided
		if dictionary_file != None:
			buffer = respell(buffer)
			# clean again
			for filter_ in filters:
				buffer = filter_(buffer)
		
		return buffer
	
	return map(clean_single, list_of_data)

		
