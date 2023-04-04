"""
605.649 Introduction to Machine Learning
Dr. Donnelly
Programming Project #03
20221016
Jacob M. Lundeen

The purpose of this assignment is to give you a chance to get some hands-on experience learning decision
trees for classification and regression. This time around, we are not going to use anything from the module
on rule induction; however, you might want to examine the rules learned for your trees to see if they “make
sense.” Specifically, you will be implementing a standard univariate (i.e., axis-parallel) decision tree and will
compare the performance of the trees when grown to completion on trees that use either early stopping (for
regression trees) or reduced error pruning (for classification trees).

Let’s talk about the numeric attributes. There are two ways of handling them. The first involves
discretizing (binning), similar to what you were doing in earlier assignments. This is not the preferred
approach, so we ask that you avoid binning these attributes. Instead, the second and preferred approach
is to sort the data on the attribute and consider possible binary splits at midpoints between adjacent data
points. Note that this could lead to a lot of possible splits. One way to reduce that is to consider midpoints
between data where the class changes. For regression, there is no corresponding method, so you should
consider splits near the middle of the sorted range and not consider all possible.

For decision trees, it should not matter whether you have categorical or numeric attributes, but you need
to remember to keep track of which is which. In addition, you need to implement that gain-ratio criterion
for splitting in your classification trees. Go ahead and eliminate features that act as unique identifiers of the
data points.
"""
from timeit import default_timer as timer
from collections import Counter
import pandas as pd
import numpy as np
from statistics import mean
import node
import csv

eps = np.finfo(float).eps

# Function to read in the data set. For those data sets that do not have header rows, this will accept a tuple of
# column names. It is defaulted to fill in NA values with '?'.
def read_data(data, names=(), fillna=True):
    if not names:
        return pd.read_csv(data)
    if not fillna:
        return pd.read_csv(data, names=names)
    else:
        return pd.read_csv(data, names=names, na_values='?')


# The missing_values() function takes in the data set and the column name and then fills in the missing values of the
# column with the column mean. It does this 'inplace' so there is no copy of the data set made.
def missing_values(data, column_name):
    data[column_name].fillna(value=data[column_name].mean(), inplace=True)


# The cat_data() function handles ordinal and nominal categorical data. For the ordinal data, we use a mapper that maps
# the ordinal data to integers, so they can be utilized in the ML algorithms. For nominal data, Pandas get_dummies()
# function is used.
def cat_data(data, var_name='', ordinal=False, data_name=''):
    if ordinal:
        if data_name == 'cars':
            buy_main_mapper = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
            door_mapper = {'2': 2, '3': 3, '4': 4, '5more': 5}
            per_mapper = {'2': 2, '4': 4, 'more': 5}
            lug_mapper = {'small': 0, 'med': 1, 'big': 2}
            saf_mapper = {'low': 0, 'med': 1, 'high': 2}
            class_mapper = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
            mapper = [buy_main_mapper, buy_main_mapper, door_mapper, per_mapper, lug_mapper, saf_mapper, class_mapper]
            count = 0
            for col in data.columns:
                data[col] = data[col].replace(mapper[count])
                count += 1
            return data
        elif data_name == 'forest':
            month_mapper = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                            'oct': 10, 'nov': 11, 'dec': 12}
            day_mapper = {'sun': 1, 'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}
            data.month = data.month.replace(month_mapper)
            data.day = data.day.replace(day_mapper)
            return data
        elif data_name == 'cancer':
            class_mapper = {2: 0, 4: 1}
            data[var_name] = data[var_name].replace(class_mapper)
            return data
    else:
        return pd.get_dummies(data, columns=var_name, prefix=var_name)


# The discrete() function transforms real-valued data into discretized values. This function provides the ability to do
# both equal width (pd.cut()) and equal frequency (pd.qcut()). The function also provides for discretizing a single
# feature or the entire data set.
def discrete(data, equal_width=True, num_bin=20, feature=""):
    if equal_width:
        if not feature:
            for col in data.columns:
                data[col] = pd.cut(x=data[col], bins=num_bin)
            return data
        else:
            data[feature] = pd.cut(x=data[feature], bins=num_bin)
            return data
    else:
        if not feature:
            for col in data.columns:
                data[col] = pd.qcut(x=data[col], q=num_bin, duplicates='drop')
            return data
        else:
            data[feature] = pd.qcut(x=data[feature], q=num_bin)
            return data


# The standardization() function performs z-score standardization on a given train and test set. The function
# will standardize either an individual feature or the entire data set. If the standard deviation of a variable is 0,
# then the variable is constant and adds no information to the regression, so it can be dropped from the data set.
def standardization(train, test=pd.DataFrame(), feature=''):
    if test.empty:
        for col in train.columns:
            if train[col].dtype == "object":
                continue
            else:
                if train[col].std() == 0:
                    train.drop(col, axis=1, inplace=True)
                else:
                    train[col] = (train[col] - train[col].mean()) / train[col].std()
        return train
    elif not feature:
        for col in train.columns:
            if train[col].std() == 0:
                train.drop(col, axis=1, inplace=True)
                test.drop(col, axis=1, inplace=True)
            else:
                test[col] = (test[col] - train[col].mean()) / train[col].std()
                train[col] = (train[col] - train[col].mean()) / train[col].std()
        return train, test
    else:
        test[feature] = (test[feature] - train[feature].mean()) / train[feature].std()
        train[feature] = (train[feature] - train[feature].mean()) / train[feature].std()
        return train, test


# The val_split() function handles splitting and stratifying classification data. Returns validation set and
# data_splits.
def val_split(data, k, class_var, validation=False):
    # Group the data set by class variable using pd.groupby()
    grouped = data.groupby([class_var])
    grouped_l = []
    data_splits = []
    # Create stratified validation set using np.split(). 20% from each group is appended to the validation set, the rest
    # will be used for the k-folds.
    if validation:
        grouped_val = []
        grouped_dat = []
        for name, group in grouped:
            val, dat = np.split(group, [int(0.2 * len(group))])
            grouped_val.append(val)
            grouped_dat.append(dat)
        # Split the groups into k folds
        for i in range(len(grouped_dat)):
            grouped_l.append(np.array_split(grouped_dat[i], k))
    else:
        for name, group in grouped:
            grouped_l.append(np.array_split(group.iloc[np.random.permutation(np.arange(len(group)))], k))
        for i in range(len(grouped_l)):
            for j in range(len(grouped_l[i])):
                grouped_l[i][j].reset_index(inplace=True, drop=True)
        for i in range(k):
            temp = grouped_l[0][i]
            for j in range(1, len(grouped_l)):
                temp = pd.concat([temp, grouped_l[j][i]], ignore_index=True)
            data_splits.append(temp)
    # Reset indices of the folds
    for item in range(len(grouped_l)):
        for jitem in range(len(grouped_l[item])):
            grouped_l[item][jitem].reset_index(inplace=True, drop=True)
    # Combine folds from each group to create stratified folds
    for item in range(k):
        tempo = grouped_l[0][item]
        for jitem in range(1, len(grouped_l)):
            tempo = pd.concat([tempo, grouped_l[jitem][item]], ignore_index=True)
        data_splits.append(tempo)
    if validation:
        grouped_val = pd.concat(grouped_val)
    else:
        grouped_val = 0
    return grouped_val, pd.concat(data_splits)


# The reg_split() function creates the k-folds for regression data.
def reg_split(data, k, validation=False):
    # Randomize the data first
    df = data.sample(frac=1, random_state=42).reset_index(drop=True)
    # If a validation set is required, divide the data set 20/80 and return the sets
    if validation:
        val_fold, data_fold = np.split(df, [int(.2 * len(df))])
        if k == 1:
            return val_fold, data_fold.reset_index(drop=True)
        else:
            data_fold = np.array_split(data_fold, k)
            return val_fold, data_fold
    # If no validation set is required, split the data by the requested k
    else:
        data_fold = np.array_split(df, k)
        val_fold = 0
        return val_fold, data_fold


# The k2_cross() function performs Kx2 cross validation.
def k2_cross(data, pred_type, max_depth=5, theta=5):
    results = []
    count = 0
    if pred_type == 'regression':
        data = standardization(data)

    # As we loop over k, we randomize each loop and then split the data 50/50 into train and test sets (standardizing
    # when doing regression). The learning algorithm is trained on the training set first and then tested on the test
    # set. They are then flipped (trained on the test set and tested on the train set). So we get 2k experiments.
    while count < 5:
        rand_df = data.sample(frac=1, random_state=42).reset_index(drop=True)
        dfs = np.array_split(rand_df, 2)
        train = dfs[0]
        test = dfs[1]
        tree = grow_tree(train.copy(), theta=theta, max_depth=max_depth, method=pred_type)
        yhat, y_test = class_predict(tree, prune_set)
        if count == 0:
            print('The size of this test set is: ' + str(test.shape[0]))
            print('The output from this test set is: ')
            print(yhat)
        results.append(eval_metrics(y_test, yhat, pred_type))
        tree = grow_tree(test.copy(), theta=theta, max_depth=max_depth, method=pred_type)
        yhat, y_test = class_predict(tree, prune_set)
        results.append(eval_metrics(y_test, yhat, pred_type))
        count += 1
    metric1 = []
    metric2 = []
    metric3 = []
    for lst in results:
        metric1.append(lst[0])
        metric2.append(lst[1])
        metric3.append(lst[2])
    final_metrics = [mean(metric1), mean(metric2), mean(metric3)]
    return final_metrics, tree


# The reg_adjust() algorithm takes an accurately predicted example during regression and sets the predicted value to be
# equal to the true value
def reg_adjust(pred, true, epsilon):
    for i in range(len(pred)):
        if (true[i] - epsilon < pred[i]) & (pred[i] < true[i] + epsilon):
            pred[i] = true[i]
    return pred


# Calculate the R2 score
def r2_score(true, pred):
    if len(pred) == 0:
        return 0
    else:
        ss_t = 0
        ss_r = 0
        mean_true = np.mean(true)
        for i in range(len(true)):
            ss_r += (true[i] - pred[i]) ** 2
            ss_t += (true[i] - mean_true) ** 2
        if ss_t == 0:
            mse = 1
        else:
            mse = round(1 - (ss_r / ss_t), 3)
        return mse


# The eval_metrics() function returns the classification or regression metrics.
def eval_metrics(true, predicted, eval_type='regression'):
    # For regression, we create the correlation matrix and then calculate the R2, Person's Correlation, and MSE.
    if len(predicted) == 0:
        return 0, 0, 0
    elif eval_type == 'regression':
        r2_s = r2_score(true, predicted)
        persons = round(pd.Series(true).corr(pd.Series(predicted)), 3)
        mse = round(np.square(np.subtract(true, predicted)).mean(), 3)
        return r2_s, persons, mse
    # For classification, we calculate Precision, Recall, and F1 scores.
    else:
        total_examp = len(true)
        precision = []
        recall = []
        f_1 = []
        count = 0
        for label in np.unique(true):
            class_len = np.sum(true == label)
            true_pos = np.sum((true == label) & (predicted == label))
            true_neg = np.sum((true != label) & (predicted != label))
            false_pos = np.sum((true != label) & (predicted == label))
            false_neg = np.sum((true == label) & (predicted != label))
            if true_pos & false_pos == 0:
                precision.append(0)
            else:
                precision.append(true_pos / (true_pos + false_pos))
            if true_pos + false_neg == 0:
                recall.append(0)
            else:
                recall.append(true_pos / (true_pos + false_neg))
            if precision[count] + recall[count] == 0:
                f_1.append(0)
            else:
                if len(np.unique(true)) > 1:
                    f_1.append((class_len / total_examp) * 2 * (precision[count] * recall[count]) / (
                                precision[count] + recall[count]))
                else:
                    f_1.append(2 * (precision[count] * recall[count]) / (precision[count] + recall[count]))
            count += 1
        if count > 1:
            return mean(precision), mean(recall), mean(f_1)
        else:
            return sum(precision), sum(recall), sum(f_1)


# The hyper_tune() function is used to tune the hyperparameters of the decision tree
def hyper_tune(data, pred_type):
    # Create lists of values for max depth and theta and loop, over to determine parameters (lowest MSE or highest F1)
    theta = [i for i in range(1, 11)]
    max_depth = [i for i in range(3, 10)]
    if pred_type == 'regression':
        df_hyper = pd.DataFrame(columns=['Max Depth', 'Theta', 'MSE'])
        for md in max_depth:
            for t in theta:
                results = k2_cross(data=data, pred_type=pred_type, theta=t, max_depth=md)
                temp = pd.DataFrame(data={'Max Depth': md, 'Theta': t, 'MSE': round(results[2], 3)}, index=[0])
                df_hyper = pd.concat([df_hyper, temp], ignore_index=True)
        df_hyper.to_csv('abalone_hyper.csv')
        mse_min = df_hyper[df_hyper.MSE == df_hyper.MSE.min()]
        mse_min.reset_index(drop=True)
        return mse_min.iat[0, 0], mse_min.iat[0, 1], mse_min.iat[0, 2]
    else:
        df_hyper = pd.DataFrame(columns=['Max Depth', 'Theta', 'F1'])
        for md in max_depth:
            for t in theta:
                results = k2_cross(data, pred_type, theta=t, max_depth=md)
                temp = pd.DataFrame(data={'Max Depth': md, 'Theta': t, 'F1': round(results[2], 3)}, index=[0])
                df_hyper = pd.concat([df_hyper, temp], ignore_index=True)
        df_hyper.to_csv('house_f1.csv')
        f1_max = df_hyper[df_hyper.F1 == df_hyper.F1.max()]
        f1_max.reset_index(drop=True)
        return f1_max.iat[0, 0], f1_max.iat[0, 1], f1_max.iat[0, 2]


# Function to handle all pre-processing of the six data sets. Can have the data encoded if desired.
def pre_process(encode=False):
    # Read in data. Attribute names have to be hardcoded for all data sets minus the forest data set.
    ab_names = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight',
                'shell_weight', 'rings']
    abalone = read_data('abalone.data', ab_names)
    abalone['parent'] = None
    abalone['parent_branch'] = None
    abalone = move_targetVar(abalone, 'rings')
    cancer_names = ['code_num', 'clump_thick', 'unif_size', 'unif_shape', 'adhesion', 'epithelial_size', 'bare_nuclei',
                    'bland_chromatin', 'norm_nucleoli', 'mitosis', 'class']
    cancer = read_data('breast-cancer-wisconsin.data', cancer_names)
    cancer.drop('code_num', axis=1, inplace=True)
    cancer['parent'] = None
    cancer['parent_branch'] = None
    cancer = move_targetVar(cancer, 'class')
    car_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    cars = read_data('car.data', car_names)
    cars['parent'] = None
    cars['parent_branch'] = None
    cars = move_targetVar(cars, 'class')
    forest = read_data('forestfires.csv')
    forest['parent'] = None
    forest['parent_branch'] = None
    forest = move_targetVar(forest, 'area')
    forest['area'] = np.log2(forest['area'] + eps)
    house_names = ['class', 'infants', 'water_sharing', 'adoption_budget', 'physician_fee', 'salvador_aid',
                   'religious_schools', 'anti_sat_ban', 'aid_nic_contras', 'mx_missile', 'immigration',
                   'synfuels_cutback', 'edu_spending', 'superfund_sue', 'crime', 'duty_free', 'export_admin_africa']
    house = read_data('house-votes-84.data', house_names, fillna=False)
    house['parent'] = None
    house['parent_branch'] = None
    house = move_targetVar(house, 'class')
    machine_names = ['vendor', 'model', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp']
    machine = read_data('machine.data', machine_names)
    machine['parent'] = None
    machine['parent_branch'] = None
    machine.drop('model', axis=1, inplace=True)
    machine = move_targetVar(machine, 'erp')

    # Handle missing values; only data set that needs it is the cancer data set.
    missing_values(cancer, 'bare_nuclei')

    if encode:
        # Categorical Data; 6 of the 6 data sets have categorical data that need to be encoded for the ML algorithms.
        abalone = cat_data(abalone, [ab_names[0]])
        abalone = move_targetVar(abalone, 'rings')
        cancer = cat_data(cancer, var_name='class', data_name='cancer', ordinal=True)
        cars = cat_data(cars, var_name=car_names, data_name='cars', ordinal=True)
        forest = cat_data(forest, data_name='forest', ordinal=True)
        house = cat_data(house, var_name=list(house_names))
        house = move_targetVar(house, 'class_republican')
        machine = cat_data(machine, var_name=list(machine_names[0:2]))
        machine = move_targetVar(machine, 'erp')
        # machine.drop(machine.columns[[0, 1]], axis=1, inplace=True)

    return abalone, cancer, cars, forest, house, machine


# Calculate the entropy of a given feature.
def entropy_feat(data, feature):
    # Identify the class variable, then determine the unique values of the class variable and the given feature.
    Class = data.keys()[-1]
    class_Unique = data[Class].unique()
    feature_unique = data[feature].unique()
    entropy_val = 0
    # Nested loops to loop through each unique value of the feature and the class variable and calculate the
    # entropy.
    for feat_value in feature_unique:
        entropy_each = 0
        for class_Value in class_Unique:
            num = len(data[feature][data[feature] == feat_value][data[Class] == class_Value])
            den = len(data[feature][data[feature] == feat_value])
            frac = num / (den + eps)
            if frac > 0:
                entropy_each += frac * np.log2(frac)
        weightage = len(data[feature][data[feature] == feat_value])/len(data)
        entropy_val += weightage * -entropy_each
    return entropy_val


# Move target, or class, variable to end of dataframe.
def move_targetVar(data, target):
    cols = data.columns.tolist()
    cols.insert(len(cols), cols.pop(cols.index(target)))
    data = data.reindex(columns=cols)
    return data


# Calculate the entropy of the parent.
def entropy_set(data):
    Class = data.keys()[-1]
    entro_sum = 0
    for target in np.unique(data[Class]):
        num = data[Class].value_counts()[target]
        dem = len(data[Class])
        frac = num / dem
        entro_sum += frac * np.log2(frac)
    return -entro_sum


# Return the information gain
def gain(e_set, e_feat):
    return e_set - e_feat


# Find the intrinsic value of the current feature.
def IV(data, feature):
    feature_unique = data[feature].unique()
    iv = 0
    # Loop through each unique value of a feature and calculate IV.
    for value in feature_unique:
        weightage = len(data[feature][data[feature] == value])/len(data)
        iv += weightage * np.log2(weightage)
    return -iv


# Function returns the Gain Ratio
def gain_ratio(gain, iv):
    if iv == 0:
        return 0
    else:
        return gain/iv


# Function does the actual splitting of a data set based on the feature ('key') passed to it. Handles both categorical
# and numeric features.
def class_split(data, key):
    data['parent'] = key
    # If feature is categorical, split on the unique values of that feature.
    if data[key].dtypes == 'object':
        splits = [y for x, y in data.groupby(key)]
        branches = data[key].unique()
        # For each split, set the parent branch of each split.
        for df in splits:
            for branch in branches:
                if df.iloc[0][key] == branch:
                    df['parent_branch'] = branch
                else:
                    continue
    else:
        splits, branches = numeric_split(data, key)
        count = 0
        for df in splits:
            df.loc[:, 'parent_branch'] = branches[count]
            count += 1
    return [df.drop(key, axis=1) for df in splits]


# Function that calculates the MSE between the true values and the predicted values.
def mse(true, pred):
    if len(true) == 0:
        return 0
    else:
        return round(np.square(true.subtract(pred)).mean(), 3)


# Function returns the feature and its MSE that minimizes the MSE of the current data set when doing regression. Handles
# both categorical and numerical features.
def min_mse(data):
    feat_mse = {}
    # Last three columns of the data set are pre-processed to track the parent of the node, the branch of the parent it
    # belongs too, and the target variable.
    for col in data.columns[:-3]:
        col_mse = 0
        # For a categorical feature, group by unique values in that feature and then calculate the MSE for it.
        if data[col].dtype == 'object':
            dfs = [y for x, y in data.groupby(col)]
            for df in dfs:
                df_mean = df[df.columns[-1]].mean()
                col_mse += round(mse(df[df.columns[-1]], df_mean), 3)
            feat_mse.update({col: col_mse})
        # For a numerical feature, we sort the feature and then iterate down the values calculating the MSE for every
        # split. The sum of the MSE for each split is the total MSE for that feature.
        else:
            df = data[col].sort_values()
            df.reset_index(drop=True, inplace=True)
            count = 0
            while count < len(data[col]) - 1:
                avg = round((df[count] + df[count+1]) / 2, 3)
                left = data[data[col] <= avg].copy()
                if left.empty:
                    left_mean = 0
                else:
                    left_mean = round(left[left.columns[-1]].mean(), 3)
                right = data[data[col] > avg].copy()
                if right.empty:
                    right_mean = 0
                else:
                    right_mean = round(right[right.columns[-1]].mean(), 3)
                col_mse += round(mse(left[left.columns[-1]], left_mean), 3)
                col_mse += round(mse(right[right.columns[-1]], right_mean), 3)
                count += 1
            feat_mse.update({col: col_mse})
    if len(feat_mse) == 0:
        return 'None', 0
    else:
        with open('forest_mse.csv', "w") as f:
            for key in feat_mse.keys():
                f.write("%s,%s\n"%(key,feat_mse[key]))
        return min(feat_mse, key=feat_mse.get), min(feat_mse.values())


# Function splits a data set based on a numeric feature. DF1 is <= the median of the feature, DF2 is > the median of the
# feature. Function also returns the names of the branches.
def numeric_split(data, key):
    df1 = data[data[key] <= data[key].quantile()].copy()
    df2 = data[data[key] > data[key].quantile()].copy()
    median = data[key].quantile()
    branches = ['<= {}'.format(median), '> {}'.format(median)]
    return [df1, df2], branches


# Function returns the feature (when classification) that has the most entropy in the current data set.
def max_entropy(data, count):
    X = data.copy()
    entropy = {}
    info_gain = {}
    gain_set = entropy_set(X)
    for col in X.columns[:-3]:
        gain_feat = entropy_feat(X, col)
        igain = gain(gain_set, gain_feat)
        info_gain.update({col: igain})
        iv = IV(X, col)
        entropy.update({col: gain_ratio(igain, iv)})
    if len(entropy) == 0:
        return None, 0
    else:
        if count == 0:
            with open('cars_gainratio.csv', "w") as f:
                for key in entropy.keys():
                    f.write("%s,%s\n"%(key,entropy[key]))
            with open('cars_infogain.csv', 'w') as f:
                for key in info_gain.keys():
                    f.write("%s,%s\n"%(key,info_gain[key]))
        return max(entropy, key=entropy.get), max(entropy.values())


# This function creates the nodes of the tree. It takes in the current subset, plus the variable that is being split on,
# the parent node, the value of the node (Gain Ratio or MSE), level of the tree, node type, and node number.
def create_node(df, key, value, uniq_feats, parent=None, level=None, node_type=None, node_num=None):
    # If the feature being split on is categorical, the number of children for that node is equal to the number of
    # unique values in that feature.
    if df[key].dtypes == 'object':
        num_children = len(uniq_feats)
    # If the feature being split on is numerical, there is a binary split. The path to the left child is <= to the
    # median value of the feature, the path to the right child is > the median.
    else:
        num_children = 2
        median = data[key].quantile()
        uniq_feats = [median, median]
    pred = class_pred(df)
    if df.empty:
        return
    if node_type == 'Leaf':
        nd = node.Node(feature=key, gain=0, parent=parent, level=level, node_type=node_type, prediction=pred,
                       samples=df.shape[0], parent_branch=df['parent_branch'].unique()[0], node_num=node_num)
        return nd
    if num_children == 2:
        nd = node.Node(feature=key, gain=value, data_left=uniq_feats[0], data_right=uniq_feats[1], parent=parent,
                       level=level, node_type=node_type, prediction=pred, samples=df.shape[0],
                       parent_branch=df['parent_branch'].unique()[0], node_num=node_num)
    elif num_children == 3:
        nd = node.Node(feature=key, gain=value, data_left=uniq_feats[0], data_middle=uniq_feats[1],
                       data_right=uniq_feats[2], parent=parent, level=level, node_type=node_type, prediction=pred,
                       samples=df.shape[0], parent_branch=df['parent_branch'].unique()[0], node_num=node_num)
    elif num_children >= 4:
        nd = node.Node(feature=key, gain=value, data_left=uniq_feats[0], data_mleft=uniq_feats[1],
                       data_mright=uniq_feats[2], data_right=uniq_feats[3], parent=parent, level=level,
                       node_type=node_type, prediction=pred, samples=df.shape[0],
                       parent_branch=df['parent_branch'].unique()[0], node_num=node_num)
    return nd


# This function returns the prediction for a node. All target variables (both regression and classification) are
# pre-processed to be the last column of a data set.
def class_pred(df):
    # If it is classification, return the most common class.
    if df[df.keys()[-1]].dtype == 'object':
        counter = Counter(df[df.keys()[-1]])
        return counter.most_common()[0][0]
    # If it is regression, return the mean value of the data set.
    else:
        return df[df.keys()[-1]].mean()


# Function grows the decision tree. Handles both regression and classification. Early stopping is handled two ways: max
# depth of the tree and minimum number of examples needed for a split. Both are preset to 5.
def grow_tree(data, theta=5, max_depth=5, method='regression'):
    tree = []
    splits = [data]
    count = 0
    while count < max_depth:
        new_splits = []
        node_num = 0
        for df in splits:
            if method == 'classification':
                key, value = max_entropy(df.copy(), count)
                if key is None:
                    continue
                else:
                    feature_unique = df[key].unique()
            else:
                key, value = min_mse(df.copy())
                if key is None:
                    continue
                elif df[key].dtype == 'object':
                    feature_unique = df[key].unique()
                else:
                    feature_unique = key
            parent = df['parent'].unique()
            if value == 0:
                nd = create_node(df, key, value, feature_unique, parent=parent, level=count, node_type='Leaf',
                                 node_num=node_num)
                tree.append(nd)
                node_num += 1
                continue
            if count == 0:
                nd = create_node(df, key, value, feature_unique, parent=None, level=count, node_type='Root',
                                 node_num=node_num)
            elif df.shape[0] <= theta:
                nd = create_node(df, key, value, feature_unique, parent=parent, level=count, node_type='Leaf',
                                 node_num=node_num)
                tree.append(nd)
                node_num += 1
                continue
            else:
                nd = create_node(df, key, value, feature_unique, parent=parent, level=count, node_type='Decision',
                                 node_num=node_num)
                node_num += 1
            tree.append(nd)
            new_splits.append(class_split(data=df, key=key))
        count += 1
        splits = [item for sublist in new_splits for item in sublist]
    return tree


# Function to print tree to a test file.
def print_tree(tree, dname="Test"):
    file = open(str(dname) + "_tree.txt", 'w', encoding='utf-8')

    for nd in tree:
        try:
            const = int(nd.level * 4 ** 1.5)
            spaces = "-" * const
            if nd.node_type == 'Root':
                file.write("Root\n")
            else:
                file.write(f"|{spaces} Parent: {nd.parent}\n")
            if nd.node_type != 'Leaf':
                file.write(f"{' ' * const}     | Parent branch: {nd.parent_branch}\n")
                file.write(f"{' ' * const}     | Tree level: {nd.level}\n")
                file.write(f"{' ' * const}     | Node number: {nd.node_num}\n")
                file.write(f"{' ' * const}     | Node type: {nd.node_type}\n")
                file.write(f"{' ' * const}     | Feature selected: {nd.feature}\n")
                file.write(f"{' ' * const}     | Gain Ratio is: {round(nd.gain, 4)}\n")
                file.write(f"{' ' * const}     | Left branch: {nd.data_left}\n")
                file.write(f"{' ' * const}     | Left middle branch: {nd.data_mleft}\n")
                file.write(f"{' ' * const}     | Middle branch: {nd.data_middle}\n")
                file.write(f"{' ' * const}     | Right middle branch: {nd.data_mright}\n")
                file.write(f"{' ' * const}     | Right branch: {nd.data_right}\n")
                file.write(f"{' ' * const}     | Sample size: {nd.samples}\n")
                file.write(f"{' ' * const}     | Predicted class: {nd.prediction}\n")
            else:
                file.write(f"{' ' * const}     | Parent branch: {nd.parent_branch}\n")
                file.write(f"{' ' * const}     | Tree level: {nd.level}\n")
                file.write(f"{' ' * const}     | Node number: {nd.node_num}\n")
                file.write(f"{' ' * const}     | Node type: {nd.node_type}\n")
                file.write(f"{' ' * const}     | Sample size: {nd.samples}\n")
                file.write(f"{' ' * const}     | Predicted class: {nd.prediction}\n")
        except Exception:
            print()

    file.close()


# Function to make predictions based on the grown decision tree. Function traverses the tree and returns the prediction
# of the leaf node that it ends on.
def class_predict(tree, test):
    X_test = test.drop(test.columns[-3:], axis=1)
    y_test = test[test.columns[-1]]
    yhat = []
    count = 0
    for index, point in X_test.iterrows():
        print("This is the point traversing the tree: ")
        print(point)
        for node in tree:
            # print(node)
            if node is None:
                continue
            if node.level != count:
                continue
            if node.parent is None and node.node_type != 'Root':
                continue
            if node.node_type == 'Root':
                print("This node is a root node.")
                count += 1
                continue
            if type(point[node.parent]) == 'object':
                if node.parent_branch == point[node.parent]:
                    if node.data_left == point[node.feature]:
                        print("The point went down the left branch.")
                        count += 1
                        continue
                    elif node.data_mleft == point[node.feature]:
                        print("The point went down the middle left branch.")
                        count += 1
                        continue
                    elif node.data_middle == point[node.feature]:
                        print("The point went down the middle branch.")
                        count += 1
                        continue
                    elif node.data_mright == point[node.feature]:
                        print("The point went down the middle right branch.")
                        count += 1
                        continue
                    elif node.data_right == point[node.feature]:
                        print("The point went down the right branch.")
                        count += 1
                        continue
                    elif node.node_type == 'Leaf':
                        print('The point has reached a leaf node and is predicted to be: ' + str(node.prediction))
                        yhat.append(node.prediction)
                        break
                else:
                    continue
            else:
                if node.node_type == 'Leaf':
                    print('The point has reached a leaf node and is predicted to be: ' + str(node.prediction))
                    yhat.append(node.prediction)
                    break
                elif node.data_left <= point[node.feature]:
                    print("The point went down the left branch.")
                    count += 1
                    continue
                elif node.data_right > point[node.feature]:
                    print("The point went down the right branch.")
                    count += 1
                    continue
                else:
                    continue
        count = 0
    return yhat, y_test


if __name__ == '__main__':
    # This first section read in the six data sets. 5 of the 6 data sets must have their column names hardcoded
    # (the forest data set is the only one that doesn't). A tuple is created with the column names and then passed
    # to the read_data() function along with the name of the data set. The house data is the only data set that does not
    # need missing values changed to '?'.
    abalone, cancer, cars, forest, house, machine = pre_process(encode=False)
    theta = 5
    max_depth = 5
    data = cars
    dname = 'cars'
    class_var = 'class'
    pred_type = 'classification'

    # file1 = open("Results.txt", 'w', encoding='utf-8')
    # file = open(str(dname) + "_results.txt", 'w', encoding='utf-8')
    # file.write("The " + str(dname) + " data, with target variable " + class_var + ", is a " + pred_type + " problem.")

    # Classification
    print("This is the " + dname + ' data set doing ' + pred_type + '\n')
    prune_set, train_set = val_split(data, k=1, class_var=class_var, validation=True)
    # start = timer()
    # max_depth, theta, f1 = hyper_tune(prune_set, pred_type)
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nHyperparameter tuning took: " + str(time) + " minutes.")
    # file.write('\nThe hyperparameter Max Depth is tune to: ' + str(max_depth))
    # file.write('\nThe hyperparameter \u03B8 is tuned to: ' + str(theta))
    # start = timer()
    results, tree = k2_cross(data=train_set, pred_type=pred_type, theta=theta, max_depth=max_depth)
    print(results)
    print_tree(tree, dname)
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nThe Decision Tree took: " + str(time) + " minutes.")
    # file.write('\nThe Precision score is: ' + str(round(results[0], 3)))
    # file.write('\nThe Recall score is: ' + str(round(results[1], 3)))
    # file.write('\nThe F1 score is: ' + str(round(results[2], 3)) + "\n")

    # Regression
    data = forest
    dname = 'forest'
    class_var = 'area'
    pred_type = 'regression'
    print("This is the " + dname + ' data set doing ' + pred_type + '\n')
    prune_set, train_set = reg_split(data, 1, True)

    # start = timer()
    # max_depth, theta, ms = hyper_tune(prune_set, 'regression')
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nHyperparameter tuning took: " + str(time) + " minutes.")
    # file.write('\nThe hyperparameter Max Depth is tune to: ' + str(max_depth))
    # file.write('\nThe hyperparameter \u03B8 is tuned to: ' + str(theta))
    # start = timer()
    results, tree = k2_cross(data=train_set, pred_type=pred_type, theta=theta, max_depth=max_depth)
    print(results)
    print_tree(tree, dname)
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nThe Decision Tree took: " + str(time) + " minutes.")
    # file.write("\nThe R\u00b2 is: " + str(round(results[0], 3)))
    # file.write("\nThe Pearsons Correlation is: " + str(round(results[1], 3)))
    # file.write("\nThe MSE is: " + str(round(results[2], 3)) + "\n")
    #
    # file.close()
    # # Here we print the results of the Null Model using Kx2 CV
    # class_var = 'erp'
    # pred_type = 'regression'
    # data = machine
    # dname = 'Machine'
    # file1 = open("Results.txt", 'w', encoding='utf-8')
    # # file = open(str(dname) + "_results.txt", 'w', encoding='utf-8')
    # # file.write("The " + str(dname) + " data, with target variable " + class_var + ", is a " + pred_type + " problem.")
    # val_set, data_set = reg_split(data=data, k=1, validation=True)
    # val_set = standardization(val_set)
    # # start = timer()
    # # k, sigma, epsilon = hyper_tune(data=val_set, class_var=class_var, pred_type=pred_type)
    # # end = timer()
    # # time = round((end - start) / 60, 3)
    # # file.write("\nHyperparameter tuning took: " + str(time) + " minutes.")
    # k = 2
    # sigma = 0.447
    # epsilon = 0.5
    # # file.write('\nThe hyperparameter K is tune to: ' + str(k))
    # # file.write('\nThe hyperparameter \u03C3 is tuned to: ' + str(sigma))
    # # file.write('\nThe hyperparameter \u03B5 is tuned to: ' + str(epsilon) + "\n")
    # # start = timer()
    # results = k2_cross(data_set, k, class_var, pred_type, sigma, epsilon, reduce='cnn')
    # # end = timer()
    # # time = round((end - start) / 60, 3)
    # # file.write("\nKNN with no reduction took: " + str(time) + " minutes.")
    # # file.write("\nThe R\u00b2 is: " + str(round(results[0], 3)))
    # # file.write("\nThe Pearsons Correlation is: " + str(round(results[1], 3)))
    # # file.write("\nThe MSE is: " + str(round(results[2], 3)) + "\n")
    # # start = timer()
    # # results = k2_cross(data_set, k, class_var, pred_type, sigma, epsilon, reduce='cnn')
    # # end = timer()
    # # time = round((end - start) / 60, 3)
    # # file.write("\nKNN with CNN reduction took: " + str(time) + " minutes.")
    # # file.write("\nThe R\u00b2 is: " + str(round(results[0], 3)))
    # # file.write("\nThe Pearsons Correlation is: " + str(round(results[1], 3)))
    # # file.write("\nThe MSE is: " + str(round(results[2], 3)) + "\n")
    # # start = timer()
    # # results = k2_cross(data_set, k, class_var, pred_type, sigma, epsilon, reduce='enn')
    # # end = timer()
    # # time = round((end - start) / 60, 3)
    # # file.write("\nKNN with ENN reduction took: " + str(time) + " minutes.")
    # # file.write("\nThe R\u00b2 is: " + str(round(results[0], 3)))
    # # file.write("\nThe Pearsons Correlation is: " + str(round(results[1], 3)))
    # # file.write("\nThe MSE is: " + str(round(results[2], 3)) + "\n")
    #
    # class_var = 'class_republican'
    # pred_type = 'classification'
    # data = house
    # dname = 'House'
    # val_set, data_set = class_split(data, 1, class_var, True)
    # # start = timer()
    # # k = hyper_tune(val_set, class_var, pred_type)
    # # end = timer()
    # # time = round((end - start) / 60, 3)
    # # file.write("\nHyperparameter tuning took: " + str(time) + " minutes.")
    # # file.write("\nThe hyperparameter K is tune to: " + str(k) + "\n")
    # k = 6
    # # start = timer()
    # results = k2_cross(data_set, k, class_var, pred_type, reduce='enn')
    # # end = timer()
    # # time = round((end - start) / 60, 3)
    # # file.write("\nKNN with no reduction took: " + str(time) + " minutes.")
    # # file.write('\nThe Precision score is: ' + str(round(results[0], 3)))
    # # file.write('\nThe Recall score is: ' + str(round(results[1], 3)))
    # # file.write('\nThe F1 score is: ' + str(round(results[2], 3)) + "\n")
    # # start = timer()
    # # results = k2_cross(data_set, k, class_var, pred_type, reduce='cnn')
    # # end = timer()
    # # time = round((end - start) / 60, 3)
    # # file.write("\nKNN with CNN reduction took: " + str(time) + " minutes.")
    # # file.write('\nThe Precision score is: ' + str(round(results[0], 3)))
    # # file.write('\nThe Recall score is: ' + str(round(results[1], 3)))
    # # file.write('\nThe F1 score is: ' + str(round(results[2], 3)) + "\n")
    # # start = timer()
    # # results = k2_cross(data_set, k, class_var, pred_type, reduce='enn')
    # # end = timer()
    # # time = round((end - start) / 60, 3)
    # # file.write("\nKNN with ENN reduction took: " + str(time) + " minutes.")
    # # file.write('\nThe Precision score is: ' + str(round(results[0], 3)))
    # # file.write('\nThe Recall score is: ' + str(round(results[1], 3)))
    # # file.write('\nThe F1 score is: ' + str(round(results[2], 3)))
    #
    # # file.close()
