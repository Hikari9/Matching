# SECTION 1: file IO


# read from a file and return a list of lines
def read(filename):
	ans = None
	with open(filename, 'r') as f:
		lines = f.readlines()
		ans = [str(s.strip()) for s in lines]
	return ans


# write to a file with a list of lines as input
def write(data, filename = 'out.txt', verbose = False):
	file = open(filename, 'w')
	for line in data:
		file.write(line)
		file.write('\n')
	if verbose: print 'Printed to:', filename
	file.close()

	
	
	
# SECTION 2: data manipulation

# split a list of data using delimeters
# returns a list of lists based on split parameter
def split(data, delimiters = '&/'):
	
	if not isinstance(data, list):
		raise TypeError('data must be a list of strings')
	
	return [map(str.strip, re.split('['+delimiters+']', entry)) for entry in data]


# flattens a list of lists into a single list
def flatten(list_of_lists):
	result = []
	for entry in list_of_lists:
		if isinstance(entry, list):
			res.extend(flatten(entry))
		else:
			res.append(entry)
	return res

# gets a list of unique entries with a split parameter
def smash(data, delimiters = '&/'):
	return list(set(flatten(split(data, delimiters))))