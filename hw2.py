# course: TCSS555
# Homework 2
# date: 04/04/2018
# name: Kaixuan Gao
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import pandas as pd


class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children. 
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level=0):
        if self.children == {}:  # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)
     
    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}:  # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)


def calInfoGain(attr, examples, entropy):
    vals = examples[attr].unique()
    target_val = examples[target].unique()
    total = len(examples)
    gain = entropy
    for val in vals:
        count = len(examples[examples[attr] == val])
        posCount = len(examples[(examples[target] == target_val[0]) & (examples[attr] == val)])
        negCount = len(examples[(examples[target] == target_val[1]) & (examples[attr] == val)])
        p1 = posCount / count
        p2 = negCount / count
        gain = gain - count / total * (- ((0 if p1 == 0 else p1 * math.log(p1, 2)) + (0 if p2 == 0 else p2 * math.log(p2, 2))))

    return gain


def selectAttribute(examples, target, attributes):
    if len(attributes) == 1:
        return attributes[0]

    target_val = examples[target].unique()
    posCount = len(examples[examples[target] == target_val[0]])
    negCount = len(examples[examples[target] == target_val[1]])
    count = posCount + negCount
    p1 = posCount / count
    p2 = negCount / count
    entropy = - ((0 if p1 == 0 else p1 * math.log(p1, 2)) + (0 if p2 == 0 else p2 * math.log(p2, 2)))

    attr = attributes[0]
    maxInfoGain = calInfoGain(attr, examples, entropy)
    for a in attributes:
        nextGain = calInfoGain(a, examples, entropy)
        if maxInfoGain < nextGain:
            maxInfoGain = nextGain
            attr = a

    return attr


def id3(examples, target, attributes):

    target_val = examples[target].unique()
    if len(target_val) == 1:
        return DecisionNode(target_val[0])
    if len(attributes) == 0:
        val = examples[target].value_counts()
        return DecisionNode(val.keys()[0])

    A = selectAttribute(examples, target, attributes)
    root = DecisionNode(A)
    for val in examples[A].unique():
        examples_subset = examples[examples[A] == val]
        # print(examples_subset.head())
        if len(examples_subset) == 0:
            root.children[val] = DecisionNode(examples[target].value_counts().keys()[0])
        else:
            newAttr = attributes.copy()
            newAttr.remove(A)
            root.children[val] = id3(examples_subset, target, newAttr)

    return root


####################   MAIN PROGRAM ######################

# Reading input data
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train, target, attributes)
tree.display()

# Evaluating the tree on the test data
correct = 0
for i in range(0, len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i, target]):
        correct += 1
print("\nThe accuracy is: ", correct/len(test))
