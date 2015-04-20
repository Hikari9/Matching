import numpy as np

def edit_distance(a, b, eps = 1, ignore_spaces = False):

	if ignore_spaces:
		a = a.replace(' ', '')
		b = b.replace(' ', '')
	
	
	'''
		Implements k-bounded Damerau-Levenshtein (edit) distance
		via epsilon (percentage of edits). Counts inserting,
		deleting, changing, and swapping of characters.
	'''
	
	n = len(a)
	m = len(b)
	
	if n==0 or m==0: return max(n,m)
	
	# make sure second string is shorter to save space
	if n < m:
		a,b = b,a
		n,m = m,n
	
	# check first if strings are in memo
	cur, prev = range(m+1), None
	k = eps * max(n,m)
	
	for i in xrange(1, n+1):
		cur, prev, prev2 = [i], cur, prev
		for j in xrange(1, m+1):
			insert = cur[j-1] + 1
			delete = prev[j] + 1
			change = prev[j-1] + int(a[i-1] != b[j-1])
			swap = prev[j-2] + 1 if prev2 and a[i-2:i] == b[j-2:j][::-1] else change
			cur.append(min(insert, delete, change, swap))
		if min(cur) > k:
			return 1
	
	return cur[m]

def edit_ratio(a, b, eps = 1, ignore_spaces = False):
	maxlen = max(len(a), len(b))
	if maxlen == 0: return 0
	return float(edit_distance(a,b)) / maxlen
	
def edit_radius(data, eps = 1, ignore_spaces = False):
	'''
		Returns the median pairwise edit ratio.
	'''
	n = len(data)
	if n <= 1: return 0
	dist = [edit_ratio(data[i], data[j], eps, ignore_spaces) for i in xrange(n) for j in xrange(i+1, n)]
	return np.median(dist)
