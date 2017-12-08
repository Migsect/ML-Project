#!/usr/bin/python
# Imports
import sys
import json
import math
import random
import numpy as np 
import itertools
import matplotlib.pyplot as plt

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

def calculateLikelihoods(data):
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
				likelihoods[user][word] += amount + smoothing
				totals[user] += amount + smoothing

	return likelihoods, totals

def trainNaiveBayes(data):
	priors = calculatePriors(data)
	likelihoods, totals = calculateLikelihoods(data)
	return {
		'priors': priors,
		'likelihoods': likelihoods,
		'totals': totals
	}

def applyNaiveBayes(model, inputs):
	# Getting all the model stuff
	priors = model['priors']
	likelihoods = model['likelihoods']
	totals = model['totals']

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
					counts = label_counts[word] + 1 
				else: # Otherwise it will count for 0
					counts = 1
				# Calculating the actual likelihood (and not just the counts)
				likelihood = counts / total
				if likelihood == 0:
					continue
				label_sum += math.log(amount * likelihood)
			output[label] = label_sum;
		output_sorted = sorted(output.items(), key=lambda item: item[1])
		results.append([{'user': item[0], 'score': item[1]}for item in output_sorted])
	return results

def calculateTopNAccuracy(actual, expected, top_n = 0):
	"""
	Calculates the top-N accuracy for the provided data.
	"""
	total_correct = 0
	for actual_item, expected_item in zip(actual, expected):
		tops = [score['user'] for score in actual_item]
		total_correct += calculateTopNScore(tops[::-1], expected_item['user'], top_n)
	return total_correct / len(actual)

def calculatePartialAccuracy(actual, expected, coefficient = 1):
	"""
	Calculates the partial accuracy for the provided data
	"""
	total_correct = 0
	for actual_item, expected_item in zip(actual, expected):
		tops = [score['user'] for score in actual_item]
		total_correct += calculatePartialScore(tops[::-1], expected_item['user'], coefficient)
	return total_correct / len(actual)

def calculateTopNScore(ranking, target, top_n):
	"""
	Calculates the score based on if the target appears in the top N items of the ranking.
	If it appears in the top_n, then it will provide a score of 1, otherwise it will provide
	a score of 0.

	Rankings are expected to be ordered in the top ranking being first.
	"""
	return ranking.index(target) < top_n

def calculatePartialScore(ranking, target, coefficient=1):
	"""
	Calculates the Partial score where all positions contribute partially to the
	accuracy.  For example, if there are 10 possible ranking places, then if the element
	is ranked first it will provide a score of 1.  if it ranked last, then is will provide
	a score of 0.  If it ranked nth it will provide a score of (len(rankings) - n - 1) / (len(rankings) - 1)  
	
	The coefficient will raise the result to a power, dampening the result such that lower
	rankings will provide less or more score.  (more is coefficient is less than 1, less if
	coefficient is greater than 1)
	"""
	return ((len(ranking) - ranking.index(target) - 1) / (len(ranking) - 1)) ** coefficient

def calculateConfusionMatrix(actual, expected, labels):
	"""
	This function calculates the confusion matrix for the confusion and actual.
	The columns represent the predicted labels (x-axis) while the rows represent 
	the true label (y-axis).
	"""
	actual_labels = [item[-1]['user'] for item in actual]
	expected_labels = [item['user'] for item in expected]

	label_count = len(labels) # Getting a list of classes
	confusion = np.zeros([label_count, label_count])
	for actual_user, expected_user in zip(actual_labels, expected_labels):
		column_index = labels.index(actual_user)
		row_index = labels.index(expected_user)
		confusion[row_index, column_index] += 1
	return confusion

def calculateSimpleConfusion(confusion, n):
	"""
	Calculates the tp, tn, fn, fp rates of the confusion matrix for a singular row.
	"""
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for r, c in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
		if r == n and c == n:
			tp += confusion[r, c]
		elif r == n:
			fn += confusion[r, c]
		elif c == n:
			fp += confusion[r, c]
		else:
			tn += confusion[r, c]
	return tp, tn, fp, fn

def calculateRecall(confusion, n):
	"""
	Calculates the recall for the nth item in the confusion matrix.
	"""
	tp, tn, fp, fn = calculateSimpleConfusion(confusion, n)
	return tp / (tp + fp)


def calculatePrecision(confusion, n):
	"""
	Calculates the precision for the nth item in the confusion matrix.
	"""
	tp, tn, fp, fn = calculateSimpleConfusion(confusion, n)
	return tp / (tp + fn)

def calculateFMeansure(confusion, n):
	"""
	Calculates the precision for the nth item in the confusion matrix.
	"""
	precision = calculatePrecision(confusion, n)
	recall = calculateRecall(confusion, n)
	return 2 * (precision * recall) / (precision + recall)

def analyzeModel(actual, expected, subject):
	"""
	Performs analysis for the model. This will generates graphs as well as outputs various
	statistics for the model.
	"""
	print("\tTop-1 Accuracy: {}".format(calculateTopNAccuracy(actual, expected, 1)))
	print("\tTop-3 Accuracy: {}".format(calculateTopNAccuracy(actual, expected, 3)))
	print("\tTop-5 Accuracy: {}".format(calculateTopNAccuracy(actual, expected, 5)))
	print("\tPartial-10 Accuracy: {}".format(calculatePartialAccuracy(actual, expected, 10)))

	# Getting the labels
	labels = sorted([item['user'] for item in actual[0]])

	# Calculating and displaying the confusion matrix
	confusion = calculateConfusionMatrix(actual, expected, labels)
	plt.figure(figsize=(9, 6), dpi=100)
	plotConfusionMatrix(confusion, labels, title=subject + ' : Confusion matrix')

	# Calculating precision, recall, and F-measure
	precision_aggregate = 0
	recall_aggregate = 0
	f_measure_aggregate = 0
	for n in range(len(labels)):
		precision_aggregate += calculatePrecision(confusion, n)
		recall_aggregate += calculateRecall(confusion, n)
		f_measure_aggregate += calculateFMeansure(confusion, n)
	print("\n\tAverage Precision: {}".format(precision_aggregate / len(labels)))
	print("\tAverage Recall: {}".format(recall_aggregate / len(labels)))
	print("\tAverage F-Measure: {}".format(f_measure_aggregate / len(labels)))

def plotConfusionMatrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '{:.2f}' if normalize else '{:.0f}'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, fmt.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
	# Loading and splitting the data
	data_file = sys.argv[1]
	data = json.load(open(data_file, 'r', encoding='ascii'))
	training_data, testing_data = randomlySplitData(data, training_ratio)

	# Training the model
	model = trainNaiveBayes(training_data)

	# Printing the priors:
	print("\nPriors Listing:")
	for user, prior in model['priors'].items():
		print("\t{}: {}".format(user, prior))

	# Applying the model to the training data
	print("\nAnalyzing Training Data Set:")
	training_results = applyNaiveBayes(model, training_data)
	analyzeModel(training_results, training_data, "Training")

	# Applying the model to the testing data
	print("\nAnalyzing Testing Data Set:")
	testing_results = applyNaiveBayes(model, testing_data)
	analyzeModel(testing_results, testing_data, "Testing")
	
	# Going over a variety of ratios and plotting their three accuracies
	splits = np.linspace(0.1, 0.9, 20)

	top_1_accuracy_training = []
	top_5_accuracy_training = []
	partial_10_accuracy_training = []

	top_1_accuracy_testing = []
	top_5_accuracy_testing = []
	partial_10_accuracy_testing = []

	for split_ratio in splits:
		training_data, testing_data = randomlySplitData(data, split_ratio)
		training_results = applyNaiveBayes(model, training_data)
		testing_results = applyNaiveBayes(model, testing_data)

		top_1_accuracy_training.append(calculateTopNAccuracy(training_results, training_data, 1))
		top_5_accuracy_training.append(calculateTopNAccuracy(training_results, training_data, 5))
		partial_10_accuracy_training.append(calculatePartialAccuracy(training_results, training_data, 10))

		top_1_accuracy_testing.append(calculateTopNAccuracy(testing_results, testing_data, 1))
		top_5_accuracy_testing.append(calculateTopNAccuracy(testing_results, testing_data, 5))
		partial_10_accuracy_testing.append(calculatePartialAccuracy(testing_results, testing_data, 10))

	# Plotting the top-N
	plt.figure(figsize=(9, 6), dpi=100)
	top_1_training = plt.plot(splits, top_1_accuracy_training, 'b', label='Top-1 - Training')
	top_5_training = plt.plot(splits, top_5_accuracy_training, 'r', label='Top-5 - Training')
	partial_training = plt.plot(splits, partial_10_accuracy_training, 'g', label='Partial - Training')
	top_1_testing = plt.plot(splits, top_1_accuracy_testing, 'b--', label='Top-1 - Testing')
	top_5_testing = plt.plot(splits, top_5_accuracy_testing, 'r--', label='Top-5 - Testing')
	partial_testing = plt.plot(splits, partial_10_accuracy_testing, 'g--', label='Partial - Testing')
	
	plt.legend()
	plt.title("Accuracy for Various Split Ratios")
	plt.xlabel("Split Ratio (training / testing)")
	plt.ylabel("Accuracy")
	plt.ylim(0.5, 1.0)
	# Showing all then plots
	plt.show()

if __name__ == '__main__':
	main()