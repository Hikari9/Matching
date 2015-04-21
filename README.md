# Matching

To run, fork then open "Course-Industry Matching.ipynb" in ipython notebook.
All important functions are explained there.

# Analysis

This repository analyzes the likelihood of matching between two independent sets of data (e.g. Course to Industry). The algorithm performs an initial <b>Content-Based Filtering</b> through features in text, with a dynamic capability of <b>Collaborative Filtering</b> through present user profiles.

Such likelihood is quantified using a matrix, where each entry describes the relative likelihood of matching. This is ideal for it is scalable with new data, and it is compatible with multiple criteria likelihood (e.g. Course to Industry to Jobs). One just needs to multiply the respective matrices to acquire a new likelihood relationship.

# Algorithm

The steps of the algorithm is as follows:
<ol>
<li>Data Mining / Data Gathering</li><br>
<li>Data Cleaning
  <ul>
  <li>text normalization</li>
  <li>prefix removal</li>
  <li>abbreviation mapping</li>
  <li>internal respelling</li>
  </ul>
</li><br>
<li>Clustering
  <ul>
  <li>Uses <i>WORD STEMMING</i> and <i>WORD FREQUENCY</i></li>
  </ul>
</li><br>
<li>Creation of Likelihood Matrix
  <ul>
  <li>Content-based Filtering</li>
  <li>Uses cosine similarity of features</li>
  <li>Tfdif vectorization of text</li>
  </ul>
</li><br>
<li>Dynamic Update of Likelihood
  <ul>
  <li>Collaborative Filtering</li>
  <li>Uses cosine similarity as well</li>
  <li>Increases likelihood for each new user info (example below)
    <ul>
    <li>user course: MARKETING</li>
    <li>user work industry: FINANCE INDUSTRY</li>
    <li>result: likelihood match of MARKETING and FINANCE increases</li>
    </ul>
  </li>
  <li>Uses cross product of all possible keyword matches</li>
  </ul>
</li><br>
<li>Repeat of previous step (5)</li>
</ol>

# Python Requirements (through pip)
	
	1) pyenchant
		- with AbiWord Enchant 
	2) stemming
	3) numpy
	4) scipy
	5) sklearn
	6) pandas
