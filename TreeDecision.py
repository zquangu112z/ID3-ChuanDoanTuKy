import math

#find most common value for an attribute
def mostCommonValue(data, attributes, target):
    valCount = {}
    # find target in data
    index = attributes.index(target)
    # calculate frequency of values in target attr
    for record in data:
        if (valCount.has_key(record[index])):
            valCount[record[index]] += 1
        else:
            valCount[record[index]] = 1
    max = 0
    major = ""
    for key in valCount.keys():
        if valCount[key] > max:
            max = valCount[key]
            major = key
    return major


# Calculates the entropy of the given data set for the target attr
def entropy(data , attributes, targetAttr):
    valFreq = {}
    dataEntropy = 0.0

    # find index of the target attribute
    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        ++i

    # Calculate the frequency of each of the values in the target attr
    for entry in data:
        if (valFreq.has_key(entry[i])):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]] = 1.0

    # Calculate the entropy of the data for the target attr
    for freq in valFreq.values():
        dataEntropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return dataEntropy

# Calculates the information gain (reduction in entropy)
# that would result by splitting the data on the chosen attribute (attr).
def gain(data,attributes, attr, targetAttr):
    valFreq = {}
    subsetEntropy = 0.0

    # find index of the attribute
    i = attributes.index(attr)

    # Calculate the frequency of each of the values in the target attribute
    for entry in data:
        if (valFreq.has_key(entry[i])):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]] = 1.0
    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in valFreq.keys():
        valProb = valFreq[val] / sum(valFreq.values())
        dataSubset = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(dataSubset,  attributes, targetAttr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data , attributes, targetAttr) - subsetEntropy)


def chooseAttribute_has_HighestIG(data, attributes, target):
    best = attributes[0]
    maxGain = 0;
    for attr in attributes:
        newGain = gain(data, attributes, attr, target)
        if newGain > maxGain:
            maxGain = newGain
            best = attr
    return best

#List of value's name of the attribute 'best'
def getValues(data, attributes, best):
    index = attributes.index(best)
    values = []
    for record in data:
        if record[index] not in values:
            values.append(record[index])
    return values

#Loai bo column best, loai bo row khong co' val
#=> new data table (step 5)
def createReduceTabble(data, attributes, best, val):
    reducedTable = [[]]
    index = attributes.index(best)

    for record in data:
        # find entries with the give value
        if (record[index] == val):
            newEntry = []
            # add value if it is not in best column
            for i in range(0, len(record)):
                if (i != index):
                    newEntry.append(record[i])
            reducedTable.append(newEntry)
    reducedTable.remove([])
    return reducedTable



def makeTree(data, attributes, target):

    data = data[:]
    target_column = [record[attributes.index(target)] for record in data]

    # If dataset is empty OR attributes list is empty Then return the most common value
    if not data or (len(attributes)-1) <= 0:
        return mostCommonValue(data, attributes, target)
    # if all the records have the same clssification Then return that classification
    elif target_column.count(target_column[0]) == len(target_column):
        return target_column[0]
    else:
        # Choose the next best attribute have highest IG
        best = chooseAttribute_has_HighestIG(data, attributes, target)

        # build a dictionary
        tree = {best:{}}

        # sub_tree
        values = getValues(data, attributes, best)
        for val in values:
            newData = createReduceTabble(data, attributes, best, val)
            newAttributes = attributes[:]
            newAttributes.remove(best)

            subtree = makeTree(newData,newAttributes,target)

            tree[best][val] = subtree

    return tree