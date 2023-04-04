import pandas as pd
import numpy as np
from collections import Counter


class cart_Regress():

    def __init__(self,
                 y: list,
                 X: pd.DataFrame,
                 min_samples_split=None,
                 max_depth=None,
                 depth=None,
                 node_type=None,
                 rule=None
                 ):

        self.y = y
        self.X = X
        self.min_samples_split = min_samples_split if min_samples_split else 15
        self.max_depth = max_depth if max_depth else 2
        self.depth = depth if depth else 0
        self.node_type = node_type if node_type else 'root'
        self.features = list(self.X.columns)
        self.rule = rule if rule else ""
        self.ymean = np.mean(y)
        self.residuals = self.y - self.ymean
        self.mse = self.get_mse(y, self.ymean)
        self.n = len(y)
        self.left = None
        self.right = None
        self.best_features = None
        self.best_value = None

    @staticmethod
    def get_mse(y_true, y_hat):
        n = len(y_true)
        res = y_true - y_hat
        r2 = res ** 2
        r2 = np.sum(r2)
        return r2 / n

    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list.
        """
        return np.convolve(x, np.ones(window), 'valid') / window

    def best_split(self) -> tuple:
        """
        Given the X features and Y targets calculates the best split
        for a decision tree
        """
        # Creating a dataset for spliting
        df = self.X.copy()
        df['Y'] = self.y

        # Getting the GINI impurity for the base input
        mse_base = self.mse

        # Finding which split yields the best GINI gain
        # max_gain = 0

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Getting the left and right ys
                left_y = Xdf[Xdf[feature] < value]['Y'].values
                right_y = Xdf[Xdf[feature] >= value]['Y'].values

                # Getting the means
                left_mean = np.mean(left_y)
                right_mean = np.mean(right_y)

                # Getting the left and right residuals
                res_left = left_y - left_mean
                res_right = right_y - right_mean

                # Concatenating the residuals
                r = np.concatenate((res_left, res_right), axis=None)

                # Calculating the mse
                n = len(r)
                r = r ** 2
                r = np.sum(r)
                mse_split = r / n

                # Checking if this is the best split so far
                if mse_split < mse_base:
                    best_feature = feature
                    best_value = value

                    # Setting the best gain to the current one
                    mse_base = mse_split

        return (best_feature, best_value)

    def grow_tree(self):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data
        df = self.X.copy()
        df['Y'] = self.y

        # If there is GINI to be gained, we split further
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            # Getting the best split
            best_feature, best_value = self.best_split()

            if best_feature is not None:
                # Saving the best split to the current node
                self.best_feature = best_feature
                self.best_value = best_value

                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature] <= best_value].copy(), df[
                    df[best_feature] > best_value].copy()

                # Creating the left and right nodes
                left = cart_Regress(
                    left_df['Y'].values.tolist(),
                    left_df[self.features],
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                )

                self.left = left
                self.left.grow_tree()

                right = cart_Regress(
                    right_df['Y'].values.tolist(),
                    right_df[self.features],
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                )

                self.right = right
                self.right.grow_tree()

    def print_info(self, width=4):
        """
        Method to print the information about the tree
        """
        # Defining the number of spaces
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const

        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | MSE of the node: {round(self.mse, 2)}")
        print(f"{' ' * const}   | Count of observations in node: {self.n}")
        print(f"{' ' * const}   | Prediction of node: {round(self.ymean, 3)}")

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info()

        if self.left is not None:
            self.left.print_tree()

        if self.right is not None:
            self.right.print_tree()
