# SECTION 1: file IO


# read from a file and return a list of lines
def read(filename):
	ans = None
	with open(filename, 'r') as f:
		lines = f.readlines()
		ans = [str(s).strip('\n') for s in lines]
	return ans


# write to a file with a list of lines as input
def write(data, filename = 'out.txt', verbose = False):
	file = open(filename, 'w')
	for line in data:
		file.write(line)
		file.write('\n')
	if verbose: print 'Printed to:', filename
	file.close()
	

# acquire a list of approved words from a list of data
import enchant
def approved_words(list_of_data, length_at_least = 5, dictionary = enchant.Dict('en-US')):
	approved = set()
	for row in split(list_of_data, ' '):
		for word in row:
			if word.isalpha() and len(word) >= length_at_least and dictionary.check(word):
				approved.add(word)
	return approved
	
	
# SECTION 2: data manipulation

# split a list of data using delimeters
# returns a list of lists based on split parameter
def split(data, delimiters = '&/'):
	import re
	if not isinstance(data, list):
		raise TypeError('data must be a list of strings')
	
	return [map(str.strip, re.split('['+delimiters+']', entry)) for entry in data]


# flattens a list of lists into a single list
def flatten(list_of_lists):
	result = []
	for entry in list_of_lists:
		if isinstance(entry, list):
			result.extend(flatten(entry))
		else:
			result.append(entry)
	return result

# gets a list of unique entries with a split parameter
def smash(data, delimiters = '&/'):
	return list(set(flatten(split(data, delimiters))))