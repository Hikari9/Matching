from collections import defaultdict
import numpy as np

'''
	A SuffixArray concatenates all data into one big string
	and sorts arbitrary features without the cost of space and time.
	
	How to use:
	
		from suffix import SuffixArray
	
		data = ['gattaca', 'atta', 'aca', 'tac']
		sa = SuffixArray(data, ignore_spaces=False)

		print '%s %s %25s\t   %10s [ID]\n' % ('Index', 'LCP', 'Concatenated Suffix', 'Original String')

		for index in sa:
			print '[%-2d]' % index,
			print '  %d' % sa.lcp[index],
			print '\t%-25s' % sa.string[index:],

			string_index = sa.G[index]
			string_data = sa.data[string_index]

			print '\t%10s [%d]' % (string_data, string_index)
			
'''
class SuffixArray(list):

	def __init__(self, strings, ignore_spaces = False, key = None):
		
		
		if key != None and isinstance(key, slice):
			# sliced constructor
			# creates a sliced copy of suffix array
			# slices only the list but not the references
			copy = strings
			list.__init__(self, list(copy).__getitem__(key))
			self.lcp = copy.lcp
			self.G = copy.G
			self.string = copy.string
			return
		
		# build long concatenated 'string' 
		S = []
		G = []
		n = len(strings)
		
		for i, s in enumerate(strings):
			if ignore_spaces: s = s.replace(' ', '')
			S.extend(map(ord, s))
			S.append(n-i-1)
			G.extend([i] * (len(s) + 1))
		
		# build main suffix array
		
		N = len(S)
		sa = range(N)
		pos = S[:]
		gap = 1
		
		def check(boolean):
			return -1 if boolean else 1
		
		def suffix_cmp(i, j):
			if pos[i] != pos[j]:
				return check(pos[i] < pos[j])
			i += gap
			j += gap
			return cmp(pos[i], pos[j]) if (i < N and j < N) else cmp(j, i)
		
		tmp = [0] * N
		
		
		while True:
			# resort using gap
			# increases gap by a factor of two
			# ensures O(nlogn) complexity
			sa.sort(cmp = suffix_cmp)
			for i in xrange(N-1):
				tmp[i+1] = tmp[i] + int(suffix_cmp(sa[i], sa[i+1]) < 0)
			for i in xrange(N):
				pos[sa[i]] = tmp[i]
			if tmp[N-1] == N-1: break
			gap <<= 1
		
		# build lowest common prefix
		lcp = [0] * N
		k = 0
		for i in xrange(N):
			if pos[i] != 0:
				j = sa[pos[i] - 1]
				while S[i+k] == S[j+k]:
					k += 1
				lcp[i] = k
				if k > 0: k -= 1
		
		# assign attributes
		list.__init__(self, sa)
		self.data = strings
		self.lcp = lcp
		self.G = G
		self.string = '$'.join((strings if not ignore_spaces else [s.replace(' ', '') for s in strings]) + [''])
	
	def __getitem__(self, key):
		# slicer
		if isinstance(key, slice):
			return SuffixArray(self, key=key)
		return list.__getitem__(self, key)
	
	def __getslice__(self, i, j):
		# override list's slice
		return self.__getitem__(slice(i, j))
	
	def connectivity(self, eps):
		# returns a list of intervals where
		# the least common prefix is at least eps
		prev = None
		x,y = 0,-1
		res = []
		for i, id in enumerate(self):
			if self.lcp[id] >= eps:
				y = i
			elif y >= x:
				res.append(slice(x-1, y+1))
				x,y = i,-1
			else:
				x = i+1
		if y != -1:
			res.append(slice(x, y+1))
		return res
		
	def similarity_graph(self):
		# returns a quantified list of weighted edges
		# where two groups are connected via LCP
		# note: the weight of an edge is calculated by sum(2 ^ lcp) for all pairs
		# O(n) complexity
		n = len(self.data)
		m = defaultdict(int)
		prev = None
		for i in self:
			if self.lcp[i] > 0:
				a = self.G[i]
				b = self.G[prev]
				if a == b: continue # don't count self
				if a > b: a, b = b, a
				c = (1 << self.lcp[i])
				m[a, b] += c
			prev = i
		return m.items()
	
	def plot(self, eps = 4):
		
		import matplotlib.pyplot as plt
		import networkx as nx
		eps = 2 ** eps
		
		G = nx.Graph()
		data = self.data
		sm = self.similarity_matrix()
		emap = defaultdict(list)
		for edge in sm.items():
			a = data[edge[0][0]]
			b = data[edge[0][1]]
			w = edge[1]
			emap[w].append((a, b))
			G.add_edge(a, b, weight = w)
		
		pos = nx.spring_layout(G)
		
		# nodes
		nx.draw_networkx_nodes(G,pos,node_size=1)
		
		from math import log
		
		# edges
		for e in emap.items():
			print 'plotting %d (%d items)' % (e[0], len(e[1]))
			nx.draw_networkx_edges(G,pos,edgelist=e[1],width=1,alpha=log(e[0]) / 32)
		
		plt.axis('off')
		plt.show()		
	
# class Trie
# By Steve Hanov, 2011. Released to the public domain
# Adopted from http://stevehanov.ca/blog/index.php?id=114

# The Trie data structure keeps a set of words, organized with one node for
# each letter. Each node has a branch for each letter that may follow it in the
# set of words.

# This data structure can search for similar words using Damerau-Levenshtein
# or edit distance. (See wikipedia for more information)

class Trie(object):

	def __init__(self, data = None, case_sensitive = False, ignore_spaces = True):
		self.word = None
		self.children = {}
		self.case_sensitive = case_sensitive
		self.ignore_spaces = ignore_spaces
		if data != None:
			if isinstance(data, Trie):
				# make a copy of trie
				self.word = data.word
				self.children = {letter: Trie(trie) for letter, trie in data.children.items()}
			else:
				for entry in data:
					self.insert(entry)
	
	def insert(self, word):
	
		node = self
		originalword = word
		
		if not self.case_sensitive:
			word = word.upper()
		
		if self.ignore_spaces:
			word = word.replace(' ', '')
		
		# letters are stored as ASCII integers (faster mapping)
		for letter in map(ord, word):
			if letter not in node.children:
				node.children[letter] = Trie()
			node = node.children[letter]
		
		node.word = originalword
	
	def __contains__(self, word):
		
		node = self
		
		if not self.case_sensitive:
			word = word.upper()
		
		if self.ignore_spaces:
			word = word.replace(' ', '')
		
		for letter in word:
			letter = ord(letter)
			if letter not in node.children:
				return False
			node = node.children[letter]
		
		return node.word != None
	
	def neighbor_iterator(self, word):
		# returns nearest neighbor
		# to use: it = trie.neighbor_iterator(word)
		#		first = it.next()
		#		second = it.next()
		# to force stop: it.send('stop')
		
		if not self.case_sensitive:
			word = word.upper()
		
		if self.ignore_spaces:
			word = word.replace(' ', '')
		
		# convert word to ASCII letters first
		word = map(ord, word)
		n = len(word)
		
		# import heap for Dijkstra / A* search
		from heapq import heappush, heappop
		queue = []
		
		# sequence: (distance, node, has_spread, is_end, prevletter, prev, prev2)
		cur = range(n + 1)
		heappush(queue, (0, self, False, False, None, None, cur))
		# heappush(queue, (n + 1, True, self, None, None, cur))
		
		
		while len(queue) > 0:
		
			distance, node, has_spread, is_end, prevletter, prev2, prev = heappop(queue)
			if is_end and node.word != None and len(node.word) != 0:
				check = yield (node.word, distance)
				if check == 'stop': return
			
			if not has_spread:
				
				# add children
				for letter in node.children:
					# build memo row first
					cur = [prev[0] + 1]
					for j in xrange(1, n + 1):
						insertCost = cur[j - 1] + 1
						deleteCost = prev[j] + 1
						replaceCost = prev[j - 1] + int(word[j - 1] != letter)
						# swapping cost: for Damerau-Levenshtein's interchanging of two letters
						if prevletter != None and j > 1 and word[j - 2] == letter and word[j - 1] == prevletter:
							swapCost = prev2[j - 2]
						else:
							swapCost = replaceCost
						cur.append(min(insertCost, deleteCost, replaceCost, swapCost))
						
					# add new possibilities to queue
					nextnode = node.children[letter]
					minimum = min(cur)
					
					heappush(queue, (minimum, nextnode, False, minimum == cur[-1], letter, prev, cur))
					
					if minimum != cur[-1]:
						heappush(queue, (cur[-1], nextnode, True, True, letter, prev, cur))
			
	
	# The search function returns a list of all words that are less than the given
	# maximum distance from the target word
	def search(self, word, k = 3):
	
		if not self.case_sensitive:
			word = word.upper()
		
		if self.ignore_spaces:
			word = word.replace(' ', '')
	
		# convert word to ASCII letters first
		word = map(ord, word)
		
		# build first row
		cur = range(len(word) + 1)
		
		results = []

		# recursively search each branch of the trie
		for letter in self.children:
			self.searchRecursive(self.children[letter], None, letter, word,
				None, cur, results, k)

		return sorted(results, key=lambda x: x[1])

	
	# This recursive helper is used by the search function above. It assumes that
	# the previousRow has been filled in already.
	def searchRecursive(self, node, prevletter, letter, word, prev2, prev, results, k):

		columns = len(word) + 1
		cur = [prev[0] + 1]

		# Build one row for the letter, with a column for each letter in the target
		# word, plus one for the empty string at column 0
		for column in xrange(1, columns):

			insertCost = cur[column - 1] + 1
			deleteCost = prev[column] + 1
			replaceCost = prev[column - 1] + int(word[column - 1] != letter)
			
			# swapping cost: for Damerau-Levenshtein's interchanging of two letters
			if prevletter != None and column > 1 and word[column - 2] == letter and word[column - 1] == prevletter:
				swapCost = prev2[column - 2]
			else:
				swapCost = replaceCost
				
			# get minimum cost from possible operations
			cur.append(min(insertCost, deleteCost, replaceCost, swapCost))

		# if the last entry in the row indicates the optimal cost is less than the
		# maximum cost, and there is a word in this trie node, then add it.
		if cur[-1] <= k and node.word != None:
			results.append((node.word, cur[-1]))

		# if any entries in the row are less than the maximum cost, then 
		# recursively search each branch of the trie
		prevletter = letter
		if min(cur) <= k:
			for letter in node.children:
				self.searchRecursive(node.children[letter], prevletter, letter, word, prev, cur, 
					results, k)