
#!/usr/bin/python

# 1. threshClassify

def threshClassify(heightList, xThresh):
	result = []
	for i in heightList:
		if i > xThresh: 
			result.append(1)
		else: 
			result.append(0)
	return result


"""
input: list 
output: [0, 0, 1, 0, 1]

heightList = [2, 5, 8, 1, 7]
xThresh = 5
result = threshClassify(heightList, xThresh)
print(result)


"""


# 2. findAccuracy

def findAccuracy(classifierOutput, trueLabels): 

	match = []
	# 1) compare output with true label
	# and append 1 to the list if match 
	for i, l in zip(classifierOutput, trueLabels): 
		if i == l:
			match.append(1)
		else:
			next 

	# 2) count accuracy as the ratio of mismatch 
	accuracy = len(match)/len(classifierOutput)
	return accuracy


"""
classifierOutput = [1, 1, 1, 0, 1, 0, 1, 1]
trueLabels = [1, 1, 0, 0, 0, 0, 1, 1]
result = findAccuracy(classifierOutput, trueLabels)
print(result)
"""

# 3. getTraining

def getTraining(data):
	result = []
	for i in data:
		result.append(i[0:2])
	return result


"""
imput: 2*3 list 
output: [[4, 5], [1, 1]]

fullData = [[4, 5, 1, 2, 8, 3], [1, 1, 0, 0, 1, 0]]
getTraining(fullData)

"""


