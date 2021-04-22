# TODO: your reusable general-purpose functions here

import math
import numpy as np  # for checking our std works

# warm up task

def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []

    for row in table:
        if (row[col_index] != "NA"):
            col.append(row[col_index])
    return col


def get_min_max(values):
    # return 2 values
    # multiple values are packed into a tuple
    # tuple: an immutable list
    return min(values), max(values)


# warm up
def get_frequencies(table, header, col_name):
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
            # have seen this value beofore
            counts[-1] += 1  # ok because col is sorted

    return values, counts

def group_by(table, header, group_by_col_name):
    col = get_column(table, header, group_by_col_name)
    col_index = header.index(group_by_col_name)
    
    # we need the unique values for our group by column
    group_names = sorted(list(set(col))) # e.g. 74, 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], [], []]
    
    # algorithm: walk through each row and assign it to the appropriate subtable
    # based on its group_by_col_name value (modelyear)
    for row in table:
        group_by_value = row[col_index]
        # which subtable to put this row in?
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row)
    
    return group_names, group_subtables

def compute_equal_width_cutoffs(values, num_bins):
    # first compute the range of the values
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins
    # bin_width is likely a float
    # if your application allows for ints, use them
    # we will use floats
    # arange is range but with floating point values (takes min, max, and step)
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values))
    # optionally: might want to round
    # define cutoffs: N + 1 values that define the edges of our bins
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs

def compute_bin_frequencies(values, cutoffs):
    freqs = [0 for _ in range(len(cutoffs) - 1)]
    
    for val in values:
        if val == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= val < cutoffs[i + 1]:
                    freqs[i] += 1
    
    
    cutoffs = compute_equal_width_cutoffs(values, num_bins)
    frequencies = []
    for i in range(len(cutoffs)):
        count = 0
        for val in range(1, len(values)):
            if values[val] <= cutoffs[i]:
                count += 1
        frequencies.append(count)
    return frequencies

def compute_slope_intercept(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => b = y -mx
    b = mean_y - m * mean_x
    return m, b

def compute_euclidean_distance(v1, v2):
    isNumeric = False
    for i, val in enumerate(v1):
        if isinstance(val, float) or isinstance(v2[i], float) or isinstance(val, int) or isinstance(v2[i], int):
#         if val.isnumeric() == False or v2[i].isnumeric() == False:
            isNumeric = True
    if isNumeric:
        if len(v1) != len(v2):
            print('len v1:', len(v1))
            print('len v2', len(v2))
            print(v1)
            print(v2)
        assert len(v1) == len(v2)
        dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
        return dist
    else:
        if v1 == v2:
            return 0
        else:
            return 1

def get_oneD_array(x):
    new = []
    for row in x:
        for val in row:
            new.append(val)
    return new
    
def display_instances(title, instances, cs, actuals):
    print_title(title)
    for j, instance in enumerate(instances):
        print('instance: ', end='')
        for i in range(len(instance) - 1):
            print(instance[i], end=', ')
        print(instance[len(instance) - 1])
        print('class: ' + str(cs[j]), end=', ')
        print('actual:', actuals[j])

def print_title(title):
    print('===========================================')
    print(title)
    print('===========================================')

def classify_mpg_instances(instances):
    classified = []
    for instance in instances:
        if instance >= 45:
            classified.append(10)
        elif instance >= 37:
            classified.append(9)
        elif instance >= 31:
            classified.append(8)
        elif instance >= 27:
            classified.append(7)
        elif instance >= 24:
            classified.append(6)
        elif instance >= 20:
            classified.append(5)
        elif instance >= 17:
            classified.append(4)
        elif instance >= 15:
            classified.append(3)
        elif instance >= 14:
            classified.append(2)
        else:
            classified.append(1)
    return classified

def classify_weight_instances(instances):
    classified = []
    for instance in instances:
        if instance >= 3500:
            classified.append(5)
        elif instance >= 3000:
            classified.append(4)
        elif instance >= 2500:
            classified.append(3)
        elif instance >= 2000:
            classified.append(2)
        else:
            classified.append(1)
    return classified

def normalize(values):
    norm_vals = []
    min_val = min(values)
    max_val = max(values)
    for val in values:
        norm_val = (val - min_val) / (max_val - min_val)
        norm_vals.append(norm_val)
    return norm_vals
        
def get_random_indices(length, num):
    vals = []
    for i in range(num):
        vals.append(np.random.randint(length))
    return vals

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
    return x_train, y_train, new_col_names
            
def get_table_from_data(data, column_names, names_to_add):
    new_table = []
    for row in data:
        new_row = []
        for name in names_to_add:
            index = column_names.index(name)
            new_row.append(row[index])
        new_table.append(new_row)
    return new_table