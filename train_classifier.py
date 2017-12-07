#!/usr/bin/python
# Imports
import sys
import json
import math
import random

training_ratio = 0.9

smoothing = 1

def randomlySplitData(data, ratio):
	"""Will randomly split the data into two sections, the first being ratio of the elements and the 
	second being the 1-ratio of the elements. 
	"""
	shuffled_data = random.sample(data, len(data))
	split_index = math.floor(len(data) * ratio)
	return shuffled_data[:split_index], shuffled_data[split_index:]

def calculatePriors(data):
	"""
	Will calculate the priors of the data (that being the probability each 
	will happen if we just assumed occurrence of the document classes).  This will return
	a dictionary that contains each class name and their corresponding ratios.
	"""
	priors = {}
	for item in data:
		user = item['user']
		if user not in priors:
			priors[user] = 0
		priors[user] += 1
	for key, amount in priors.items():
		priors[key] = priors[key] / len(data)
	return priors

def calculateLikelihoods(data, multinomial=False):
	"""
	Will calculate the likelihoods of the data (that being the probability a word will
	occur in a certain class.)  This will return a dictionary that contains each class name which
	will be a key to a dictionary of words to their ratios.

	Note that the returns are the counts of all the data and the totals.
	"""
	# A mapping of users to their documents.
	user_items = {} 
	for item in data:
		user = item['user']
		if user not in user_items:
			user_items[user] = []
		user_items[user].append(item['words'])

	# Calculating the likelihoods
	likelihoods = {}
	totals = {}
	for user, items in user_items.items():
		likelihoods[user] = {}
		totals[user] = 0
		# Looping through all the data items to add stuff up
		for item in items:
			for word, amount in item.items():
				if word not in likelihoods[user]:
					likelihoods[user][word] = 0
				# Summing the likelihoods and totals.  Adding in smoothing as well since its used later.
				likelihoods[user][word] += amount + smoothing if multinomial else 1
				totals[user] += amount + smoothing if multinomial else 1

	return likelihoods, totals

def trainNaiveBayes(data, multinomial=False):
	priors = calculatePriors(data)
	likelihoods, totals = calculateLikelihoods(data, multinomial=multinomial)
	return {
		'priors': priors,
		'likelihoods': likelihoods,
		'totals': totals,
		'multinomial': multinomial
	}

def applyNaiveBayes(model, inputs):
	# Getting all the model stuff
	priors = model['priors']
	likelihoods = model['likelihoods']
	totals = model['totals']
	multinomial = model['multinomial']

	# Getting the results
	results = []
	for input_item in inputs:
		output = {}
		input_words = input_item['words']
		for label in priors: # Looping over every label 
			label_sum = math.log(priors[label])
			total = totals[label]

			# Looping through all the label counts
			label_counts = likelihoods[label]
			for word, amount in input_words.items():
				# If the word exists in the counted likelihoods, set it
				if word in label_counts:
					counts = label_counts[word] + 1 if multinomial else 1
				else: # Otherwise it will count for 0
					counts = 1 if multinomial else 0
				# Calculating the actual likelihood (and not just the counts)
				likelihood = counts / total
				if likelihood == 0:
					continue
				label_sum += math.log(amount * likelihood if multinomial else likelihood )
			output[label] = label_sum;
		output_sorted = sorted(output.items(), key=lambda item: item[1])
		results.append([{'user': item[0], 'score': item[1]}for item in output_sorted])
	return results

def analyzeModel(actual, expected):
	total_correct = 0
	for index, result in enumerate(actual):
		if result[-1]['user'] == expected[index]['user']:
			total_correct += 1
	print("Accuracy: {}".format(total_correct / len(actual)))

def main():
	# Loading and splitting the data
	data_file = sys.argv[1]
	data = json.load(open(data_file, 'r', encoding='ascii'))
	training_data, testing_data = randomlySplitData(data, training_ratio)
	
	# Training the model
	model = trainNaiveBayes(training_data, multinomial=True)

	# Applying the model to the training data
	training_results = applyNaiveBayes(model, training_data)
	analyzeModel(training_results, training_data)

	# Applying the model to the testing data
	testing_results = applyNaiveBayes(model, testing_data)
	analyzeModel(testing_results, testing_data)
	
	print("\nPriors:")
	for p, value in model['priors'].items():
		print("\t{}: {}".format(p, value))

if __name__ == '__main__':
	main()