import math 
import numpy as np # for checking our std work
import csv


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
