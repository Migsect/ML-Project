#!/usr/bin/python
# Imports
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys

from process_data import getDocuments, wordMetrics, filterLength

# Locations
data_file = './data/forum-data.json'

def main():
	# Loading the file
	data = json.load(open(data_file, 'r', encoding='ascii'))	

	# retrieving the user-text pairs
	documents = getDocuments(data)
	print("\n{} valid documents found".format(len(documents)))

	users = set()
	for d in documents:
		users.add(d['user'])
	print("\nUnique Users: {}".format(len(users)))

	# Creating a histogram of document lengths
	print("\nCreating Document Length Histogram")
	document_lengths = [sum([amount for word, amount in d['words'].items()]) for d in documents]

	plt.figure(figsize=(9, 6), dpi=100)
	plt.hist(document_lengths, 200)
	plt.yscale("log", nonposy='clip')
	plt.ylabel("Amount of Documents")
	plt.xlabel("Number of Words")
	plt.title("Document Length Histogram")

	# Filtering the documents length so that any documents below 15 length are removed
	documents = filterLength(documents) 

	# Getting the word metrics
	word_counts, word_inclusion = wordMetrics(documents)

	# Plotting a histogram of inclusion counts
	print("\nCreating Inclusion Histogram")
	plt.figure(figsize=(9, 6), dpi=100)
	plt.hist([amount for word, amount in word_inclusion.items()], 200)
	plt.yscale("log", nonposy='clip')
	plt.ylabel("Number of Words")
	plt.xlabel("Inclusion Count")
	plt.title("Document Inclusion Histogram")

	# Plotting a histogram of word counts
	print("\nCreating Occurence Histogram")
	plt.figure(figsize=(9, 6), dpi=100)
	plt.hist([amount for word, amount in word_counts.items()], 200)
	plt.yscale("log", nonposy='clip')
	plt.ylabel("Number of Words")
	plt.xlabel("Total Occurrence Amount")
	plt.title("Total Occurrence Histogram")

	# Plotting thresholds
	thresholds = np.linspace(0.0, 0.1, 100)

	# Without filtering
	print("\nCreating Threshold Plot")

	inclusion_total = sum([amount for word, amount in word_inclusion.items()])
	occurence_total = sum([amount for word, amount in word_counts.items()])
	inclusion_ratios = [float(amount) / inclusion_total for word, amount in word_inclusion.items()]
	occurence_ratios = [float(amount) / occurence_total for word, amount in word_counts.items()]
	inclusion_valid_above = [sum([r > t for r in inclusion_ratios])  for t in thresholds]
	occurence_valid_above = [sum([r > t for r in occurence_ratios])  for t in thresholds]

	plt.figure(figsize=(9, 6), dpi=100)
	plt.plot(thresholds, inclusion_valid_above, 'b', label='Inclusion Ratios Above')
	plt.plot(thresholds, occurence_valid_above, 'r', label='Occurrence Ratios Above')
	plt.legend()
	plt.title("Ratios Counts")
	plt.xlabel("Ratio Threshold")
	plt.ylabel("Amount")
	plt.yscale("log", nonposy='clip')

	# # Filtering out single occurrence/inclusion words
	# print("\nCreating Filtered Threshold Plot")

	# filter_count = 10000
	# filtered_inclusion = [amount for word, amount in word_inclusion.items() if amount > filter_count]
	# filtered_occurence = [amount for word, amount in word_counts.items() if amount > filter_count]
	# filtered_inclusion_total = sum(filtered_inclusion)
	# filtered_occurence_total = sum(filtered_occurence)
	# filtered_inclusion_ratios = [float(amount) / filtered_inclusion_total for amount in filtered_inclusion]
	# filtered_occurence_ratios = [float(amount) / filtered_occurence_total for amount in filtered_occurence]
	# filtered_inclusion_valid_above = [sum([r > t for r in filtered_inclusion_ratios]) / len(filtered_inclusion) for t in thresholds]
	# filtered_occurence_valid_above = [sum([r > t for r in filtered_occurence_ratios]) / len(filtered_occurence) for t in thresholds]
	# filtered_inclusion_valid_below = [sum([r <= t for r in filtered_inclusion_ratios]) / len(filtered_inclusion) for t in thresholds]
	# filtered_occurence_valid_below = [sum([r <=t for r in filtered_occurence_ratios]) / len(filtered_occurence) for t in thresholds]
	
	# plt.figure(figsize=(9, 6), dpi=100)
	# plt.plot(thresholds, filtered_inclusion_valid_above, 'b', label='Inclusion Ratios Above')
	# plt.plot(thresholds, filtered_occurence_valid_above, 'r', label='Occurrence Ratios Above')
	# plt.plot(thresholds, filtered_inclusion_valid_below, 'b--', label='Inclusion Ratios Below')
	# plt.plot(thresholds, filtered_occurence_valid_below, 'r--', label='Occurrence Ratios Below')
	# plt.legend()
	# plt.title("Less-than {} Filtered - Ratios Counts".format(filter_count))
	# plt.xlabel("Ratio Threshold")
	# plt.ylabel("Amount")
	# plt.locator_params(numticks=10)
	# plt.yscale("log", nonposy='clip')

	# Showing all the graphs
	plt.show()

if __name__ == '__main__':
	main()