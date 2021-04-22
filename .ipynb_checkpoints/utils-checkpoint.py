import math 
import numpy as np # for checking our std work
import csv
import myevaluation as myevaluation

gender_index = 1
age_index = 2
married_index = 5
work_index = 6
residence_index = 7
glucose_index = 8
bmi_index = 9
smoking_index = 10

import copy

def get_accuracy_and_error_rate(classifier):
    # calulate accuracy...
    total = len(classifier.X_train)
    total_correct = 0
    predictions = classifier.predict(classifier.X_train)
    for i, val in enumerate(predictions):
        if val == classifier.y_train[i]:
            total_correct += 1
    accuracy = float(total_correct/total)
    error_rate = 1 - accuracy
    
    return accuracy, error_rate

def get_accuracy_and_error_rate_stratified(knn_classifier):
    X_train_folds, X_test_folds = myevaluation.kfold_cross_validation(knn_classifier.X_train, 10)
    total = 0
    correct = 0
    predictions = []
    for i in range(len(X_train_folds)):
        curr_testing_set = []
        for val in X_test_folds[i]:
            curr_testing_set.append(knn_classifier.X_train[val])
            total += 1
        prediction = knn_classifier.predict(curr_testing_set)
#         curr_vals = []
#         for j, val in enumerate(prediction):
#             curr_vals.append(y_train[X_test_folds[i][j]])
        for val in prediction:
            predictions.append(val)
        for m, val in enumerate(prediction):
            if val == knn_classifier.y_train[m]:
                correct += 1

    accuracy = float(correct / total)
    error_rate = 1 - accuracy
    return accuracy, error_rate

def get_trains_seperated(data, col_names, y_name):
        y_name_index = col_names.index(y_name)
        x_train = []
        y_train = []
        new_col_names = []
        for row in data:
            new_row = []
            for i, val in enumerate(row):
                if i != y_name_index:
                    new_row.append(val)
                else:
                    y_train.append(val)
            x_train.append(new_row)
        for i, name in enumerate(col_names):
            if i != y_name_index:
                new_col_names.append(name)
        return x_train, y_train

def remove_id_column(dataset):
    new_data = copy.deepcopy(dataset)
    for row in new_data:
        del row[0]
    return new_data

def classify_stroke_data(dataset):
    classified_data = copy.deepcopy(dataset)
    for i, row in enumerate(classified_data):
        # classify gender
        if row[gender_index] == "Male":
            dataset[i][gender_index] = 0
        elif row[gender_index] == "Female":
            dataset[i][gender_index] = 1
        
        # classify age
        if row[age_index] <= 2:
            dataset[i][age_index] = 0
        elif row[age_index] <= 10:
            dataset[i][age_index] = 1
        elif row[age_index] <= 20:
            dataset[i][age_index] = 2
        elif row[age_index] <= 30:
            dataset[i][age_index] = 3
        elif row[age_index] <= 40:
            dataset[i][age_index] = 4
        elif row[age_index] <= 50:
            dataset[i][age_index] = 5
        elif row[age_index] <= 60:
            dataset[i][age_index] = 6
        elif row[age_index] <= 70:
            dataset[i][age_index] = 7
        elif row[age_index] <= 80:
            dataset[i][age_index] = 8
        elif row[age_index] <= 90:
            dataset[i][age_index] = 9
        else:
            dataset[i][age_index] = 10
            
        # classify ever_married
        if row[married_index] == "No":
            dataset[i][married_index] = 0
        elif row[married_index] == "Yes":
            dataset[i][married_index] = 1
                        
        # classify work_type
        if row[work_index] == "Never_worked":
            dataset[i][work_index] = 0
        elif row[work_index] == "children":
            dataset[i][work_index] = 1
        elif row[work_index] == "Private":
            dataset[i][work_index] = 2
        elif row[work_index] == "Self-employed":
            dataset[i][work_index] = 3
        elif row[work_index] == "Govt_job":
            dataset[i][work_index] = 4
            
        # classify residence_type
        if row[residence_index] == "Urban":
            dataset[i][residence_index] = 0
        elif row[residence_index] == "Rural":
            dataset[i][residence_index] = 1
            
        # classify glucose_level
        if row[glucose_index] <= 70:
            dataset[i][glucose_index] = 0
        elif row[glucose_index] <= 140:
            dataset[i][glucose_index] = 1
        elif row[glucose_index] <= 200:
            dataset[i][glucose_index] = 2
        else:
            dataset[i][glucose_index] = 3
            
        # classify bmi
        if row[bmi_index] <= 15:
            dataset[i][bmi_index] = 0
        elif row[bmi_index] <= 20:
            dataset[i][bmi_index] = 1
        elif row[bmi_index] <= 25:
            dataset[i][bmi_index] = 2
        elif row[bmi_index] <= 30:
            dataset[i][bmi_index] = 3
        elif row[bmi_index] <= 40:
            dataset[i][bmi_index] = 4
        else:
            dataset[i][bmi_index] = 5
            
        # classify smoking_status
        if row[smoking_index]  == "never smoked":
            dataset[i][smoking_index] = 0
        elif row[smoking_index] == "formerly smoked":
            dataset[i][smoking_index] = 1
        elif row[smoking_index] == "smokes":
            dataset[i][smoking_index] = 2
            
    return classified_data

def read_csv(filename):
    '''
    This function is for reading in data from the files 
    '''
    the_file = open(filename, 'r')
    the_reader = csv.reader(the_file)
    table=[]
    for row in the_reader:
        if len(row) > 0:
            table.append(row)
    the_file.close()
    return table

def remove_unknowns(column):
    for row in column:
        if "Unknown" in row:
            column.remove(row)
    return column


def get_column(table, header, col_name):
    '''
    Function that returns the contents of a column as a list
    '''
    col_index = header.index(col_name)
    col = []

    for row in table:
        if (row[col_index] != "NA"):
            col.append(row[col_index])
    return col 

def get_col_no_clean(table, header, col_name):
    '''
    Without if statement to perserve raw table
    '''
    col_index = header.index(col_name)
    col = []

    for row in table:
        col.append(row[col_index])
    return col 

def get_min_max(values):
    # return 2 values
    # multiple values are packed into a tuple
    # tuple: an immutable list
    return min(values), max(values)

def get_frequencies(table, header, col_name):
    '''
    Function from class that accepts a specific column from a table and
    returns the values and corresponding frequency of each value as a tuple 
    '''
    col = get_column(table, header, col_name)

    col.sort() 
    
    values = []
    counts = []

    for value in col:
        if value not in values:
            # haven't seen this value before
            values.append(value)
            counts.append(1)
        else:
            # have seen this value before
            counts[-1] += 1 # ok because col is sorted

    return values, counts 

def convert_to_numeric(values):
    '''
    Function that converts non numeric values to numeric
    or returns that they cannot be converted 
    '''
    # try to convert each value in values to a numeric type
    # skip over values that can't be converted
    for i in range(len(values)):
        try: 
            numeric_value = float(values[i])
            # success!
            values[i] = numeric_value
        except ValueError:
            print(values[i], "cannot be converted")

def clean_up_col(column):
    '''
    This function is for getting rid of NA values in a column
    '''
    for row in column:
        if "NA" in row:
            column.remove(row)
    return column

def group_by(table, header, group_by_col_name):
    '''utils.py
    
    '''
    col = get_column(table, header, group_by_col_name)
    col_index = header.index(group_by_col_name)

    # get a list of unique values for the column
    group_names = sorted(list(set(col))) # 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], []]

    # walk through each row and assign it to the appropriate
    # subtable based on its group by value (model year)
    for row in table:
        group_value = row[col_index]
        # which group_subtable??
        group_index = group_names.index(group_value)
        group_subtables[group_index].append(row.copy()) # shallow copy

    return group_names, group_subtables


def compute_equal_cutoffs(values, bin_num):
    '''
    class
    '''
    convert_to_numeric(values)
    range = max(values) - min(values)
    #values_min, values_max = get_min_max(values)
    bin_size = (range / bin_num)
    
    cutoffs = list(np.arange(min(values), max(values),bin_size))
    cutoffs.append(max(values))
    
    return cutoffs


def compute_bin_frequencies(values, cutoffs):
    '''
    From class
    jupyternotebookfuns2 utils.py
    computing how many are in each bin
    '''
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for val in values:
        if val == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= val < cutoffs[i + 1]:
                    freqs[i] += 1

    return freqs
