import numpy as np
import pandas as pd
from numpy import log2 as log
eps = np.finfo(float).eps

class Regressor(object):

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.tree = None


    # Calculate the entropy of the feature.
    @staticmethod
    def entropy_set(data):
        Class = data.keys()[-1]
        entro_sum = 0
        values = data[Class].unique()
        for val in values:
            frac = data[Class].value_counts()[val] / len(data[Class])
            entro_sum += -frac * np.log2(frac)
        return entro_sum

    # Calculate the entropy of a given feature.
    @staticmethod
    def entropy_feat(data, feature):
        # Identify the class variable, then determine the unique values of the class variable and the given feature.
        Class = data.keys()[-1]
        class_Unique = data[Class].unique()
        feature_unique = data[feature].unique()
        entropy_val = 0
        # Nested loops so as to loop through each unique value of the feature and the class variable and calculate the
        # entropy.
        for feat_value in feature_unique:
            entropy_each = 0
            for class_Value in class_Unique:
                num = len(data[feature][data[feature] == feat_value][data[Class] == class_Value])
                den = len(data[feature][data[feature] == feat_value])
                frac = num / (den + eps)
                entropy_each += -frac * log(frac + eps)
            frac2 = den / len(data)
            entropy_val += -frac2 * entropy_each

        return abs(entropy_val)

    # Return the information gain
    @staticmethod
    def info_gain(e_set, e_feat):
        return e_set - e_feat

    # Find the intrinsic value of the current feature.
    @staticmethod
    def IV(data, feature):
        feature_unique = data[feature].unique()
        iv = 0
        # Loop through each unique value of a feature and calculate IV.
        for value in feature_unique:
            Ex = data[feature].value_counts()[value]
            iv += (Ex / len(data)) * log(Ex / len(data))
        return -iv

    # Function to protect against division by zero
    @staticmethod
    def divide_zero(x, y):
        return x / y if y else 0

    # Find the max entropy of the remaining features to determine splitting.
    @staticmethod
    def find_max_ent(data):
        gain = []
        # Loop through all features except the class variable
        for key in data.keys()[:-1]:
            gain.append(divide_zero(info_gain(entropy_set(data), entropy_feat(data, key)), IV(data, key)))
        return data.keys()[:-1][np.argmax(gain)]

    # Return subset of data based on current splitting node and unique value
    @staticmethod
    def get_subset(self, data, node, value):
        return data[data[node] == value].reset_index(drop=True)

    # Function to grow ID3 tree.
    def ID3(self, data, tree=None):
        # Here, we identify the class variable. All six data sets are setup (either as they were or through the
        # pre-processing function) to have the target variable as the last variable in the dataframe). The splitting node
        # is then found by determining the max entropy, and the unique values are found for that feature.
        Class = data.keys()[-1]
        node = find_max_ent(data)
        feat_Value = np.unique(data[node])
        # If tree is empty, create first node
        if tree is None:
            tree = {node: {}}
        # For each unique value in feat_Value, we subset the data set by that value and feature (node) using the
        # get_subset() function.
        for val in feat_Value:
            sub_Set = get_subset(data, node, val)
            # Find the unique class values and their counts for the sub_Set
            cl_Value, counts = np.unique(sub_Set[Class], return_counts=True)
            # If counts == 1, the leaf is equal to the current class value. Else recursively continue to build the tree.
            if len(counts) == 1:
                tree[node][val] = cl_Value[0]
            else:
                tree[node][val] = ID3(sub_Set)
        return tree

    # ID3 function to iterate over all data points in passed data set. Each data point is passed to the ID3_predict()
    # function, which computes the prediction and returns it. Once all predictions for data set are finished, ID3_fun()
    # returns list of predictions.
    def ID3_fun(self, tree, data):
        res = []
        for _, e in data.iterrows():
            res.append(ID3_predict(tree, e))
        return res

    # ID3 predictor, receives data point from ID3_fun(), traverses current tree and makes prediction.
    @staticmethod
    def ID3_predict(tree, data):
        current_node = list(tree.keys())[0]
        # try except statement to handle any values that may not be present in the tree since the tree is not grown
        # on the entire data set, but k-folds, and it is likely that not every outcome is included in the tree.
        try:
            current_branch = tree[current_node][data[current_node]]
            # if leaf node value is string or integer, then its a decision
            if isinstance(current_branch, (str, np.integer)):
                return current_branch
            # else use that node as new searching subtree
            else:
                return ID3_predict(current_branch, data)
        except KeyError:
            return current_node