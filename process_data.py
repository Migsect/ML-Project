#!/usr/bin/python

# Imports
import json
import matplotlib.pyplot as plt
import re
import os
import sys

# Options
top_users = 10

# Locations
data_file = './data/forum-data.json'

# Converts the data into a (user, words[]) tuple list
def getDocuments(data):
	documents = []
	print("Processing Documents:")
	for index, d in enumerate(data):
		# Grabbing the user from the data
		if d['user'] is None or d['text'] is None:
			continue

		# Decoding from UTF and ignoring UTF characters
		user = d['user'].encode('ascii','ignore').decode('utf-8')
		text = d['text'].encode('ascii','ignore').decode('utf-8')
		
		# Exlcuding punctuation, setting to lowercase, and setting all whitespace to spaces
		text = re.sub('[^\w\s]','', text).lower()
		
		# SPlitting the text into words
		text_split = text.split()
		words = {}
		for word in text_split:
			if word not in words:
				words[word] = 0
			words[word] += 1

		# Adding to the documents
		documents.append({'user': user, 'words': words})
		if len(documents) % 1000 == 0:
			print("\tProcessed {}/{}".format(index,  len(data)))
			sys.stdout.flush()
	
	return documents

def wordMetrics(documents):
	word_counts = {} # How many times does the word show up
	word_inclusion = {} # How many documents include the words 
	for document in documents:
		for word, amount in document['words'].items():
			if(word not in word_inclusion or word not in word_counts):
				word_counts[word] = 0
				word_inclusion[word] = 0
			word_inclusion[word] += 1
			word_counts[word] += amount
	
	return word_counts, word_inclusion

def filterWords(documents, word_inclusion, min_inclusion_ratio=0.01, max_inclusion_ratio=0.1):
	remove_words = set()
	for (word, docs) in word_inclusion.items():
		inclusion_ratio = docs / len(documents) # The ratio of documents the word appears in.
		if inclusion_ratio > max_inclusion_ratio or inclusion_ratio < min_inclusion_ratio:
			remove_words.add(word) # If the word does not meet our requirements, we'll remove it
	
	print("\nFiltering Documents of {} Excluded Words:".format(len(remove_words)))
	for index, word in enumerate(remove_words): # Looping through all the words to remove
		if index % 1000 == 0:
			print("\tProcessed {}/{} words".format(index, len(remove_words)))
			sys.stdout.flush()
		for document in documents: # Looping through all the documents
			words = document['words']
			if word in words:
				del words[word]
	return documents, remove_words

def filterLength(documents, min_words=15):
	"""
	@brief      Filters the documents with words less than the limit amount
	
	@param      documents  The documents
	
	@return     A list of documents that contain no less than min_words words.
	"""
	return [d for d in documents if sum(amount for w, amount in d['words'].items()) >= min_words]

def main():
	# Loading the file
	data = json.load(open(data_file, 'r', encoding='ascii'))	

	# retrieving the user-text pairs
	documents = getDocuments(data[:5000])
	print("\n{} valid documents found".format(len(documents)))

	# Getting the word metrics
	word_counts, word_inclusion = wordMetrics(documents)
	
	# Getting rid of documents we know to not be used.
	documents= filterLength(documents) 

	# Filtering the documents of the bad words
	documents, remove_words = filterWords(documents, word_inclusion)
	print("\nTotal amount of words: {}".format(len(word_counts) - len(remove_words)))

	# Filtering the list of documents who now may have less than the word threshold
	documents = filterLength(documents)

	# Sorting users based on their number of documents
	user_counts = {}
	for d in documents:
		user = d['user']
		if user not in user_counts:
			user_counts[user] = 0
		user_counts[user] += 1
	user_counts_sorted = sorted(user_counts.items(), key=lambda user_count: user_count[1])
	print("\nTop {} Users".format(top_users))
	for user_count in user_counts_sorted[-top_users:]:
		print("\t{}: {}".format(user_count[0], user_count[1]))

	documents = [d for d in documents if d['user'] in [user[0] for user in user_counts_sorted]]

	print('\nSaving {} documents'.format(len(documents)))

	if not os.path.exists('./processed'):
		os.makedirs('./processed')
	output_file = './processed/top{}_amount{}.json'.format(top_users, len(documents))
	with open(output_file, 'w+') as outfile:
		json.dump(documents, outfile, indent=4)
	print('\nSaved the file to {}'.format(output_file))


if __name__ == '__main__':
	main()