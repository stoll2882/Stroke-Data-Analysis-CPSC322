import copy
import csv
# from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        N = len(self.data)
        M = len(self.column_names)
        return N, M  # TODO: fix this

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = []
        try:
            col_index = self.column_names.index(col_identifier)
            for row in self.data:
                if (row[col_index] == "NA"):
                    if (include_missing_values == True):
                        col.append(row[col_index])
                else:
                    col.append(row[col_index])

        except ValueError:
            print(col_identifier, "is not a valid column name")

        return col  # TODO: fix this

    def classify_column(self, col_name):
        col_index = self.column_names.index(col_name)
        unique_vals = []
        for row in self.data:
            if row[col_index] not in unique_vals:
                unique_vals.append(row[col_index])
        for i, row in enumerate(self.data):
            self.data[i][col_index] = unique_vals.index(row[col_index])

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        rows, cols = self.get_shape()

        for i in range(rows):
            for j in range(cols):
                try:
                    numeric_value = float(self.data[i][j])
                    # success!
                    self.data[i][j] = numeric_value
                except ValueError:
                    pass
                    # print(self.data[i][j], "is not numeric")

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        rows, cols = self.get_shape()

        num_rows = rows
        i = 0
        while(i < num_rows):
            row = self.data[i]
            for j in range(len(rows_to_drop)):
                if (row == rows_to_drop[j]):
                    self.data.pop(i)
                    num_rows -= 1
            i += 1

    # function to read information from csv file into a table
    def load_from_file(self, filename):
        """Function to read in a csv file into a table

        Args:
            string: filename of csv

        Returns:
            list of two lists: the header and the table found in the csv
        """
        # create table
        table = []
        # open file for reading
        infile = open(filename, "r")
        # read lines from file
        lines = infile.readlines()
        # get the header line, strip the newline characters and split it into values
        header_line = lines.pop(0).strip()
        header = header_line.split(",")
        # print it for debugging purposes
        print("header: ", header)
        # loop through remaining lines
        for line in lines:
            # stip of newline characters
            line = line.strip()
            # get the values seperated by commas
            values = line.split(",")
            # Special case, field includes
            # Handle double quotes
            if values[1].startswith('"'):
                values = self.fix_columns_with_quotes(values)
            for i in range(len(values)):
                if values[i] == '':
                    values[i] = 'NA'
            # add formatted row to table
            table.append(values)
        self.column_names = header
        self.data = table
        return self

    # function to read information from csv file into a table
    def load_movie_data_from_file(self, filename):
        """Function to read in a csv file into a table

        Args:
            string: filename of csv

        Returns:
            list of two lists: the header and the table found in the csv
        """
        # create table
        table = []
        # open file for reading
        infile = open(filename, "r")
        # read lines from file
        lines = infile.readlines()
        # get the header line, strip the newline characters and split it into values
        header_line = lines.pop(0).strip()
        header = header_line.split(",")
        # print it for debugging purposes
        print("header: ", header)
        # loop through remaining lines
        for line in lines:
            # stip of newline characters
            line = line.strip()
            # get the values seperated by commas
            values = line.split(",")
            # Special case, field includes
            # Handle double quotes
            for i in range(len(values)):
                if values[i].startswith('"'):
                    # print(values)
                    values = self.fix_columns_with_quotes(values)
                    break
                if values[i] == '':
                    values[i] = 'NA'
            # add formatted row to table
            table.append(values)
        self.column_names = header
        self.data = table
        return self

    def fix_columns_with_quotes(self, sourceTable):
        """Reformats any colums that have quotes in them

        Args:
            list of list: original source table

        Returns:
            list of list: The new table reformat
        """
        result = []
        targetColumn = -1
        capturing = False
        capture_count = 0
        for sourceColumn in range(len(sourceTable)):
            # Remove double quotes
            sourceTable[sourceColumn] = sourceTable[sourceColumn].replace(
                '""', '"')
            if sourceTable[sourceColumn].startswith('"'):
                # Handle special case of start and end with quote
                if sourceTable[sourceColumn].endswith('"'):
                    result.append(sourceTable[sourceColumn]
                                  [1:len(sourceTable[sourceColumn]) - 1])
                    targetColumn += 1
                else:
                    result.append(sourceTable[sourceColumn].lstrip('"'))
                    result[targetColumn] = result[targetColumn] + ","
                    capturing = True
                    targetColumn = sourceColumn - capture_count
            elif capturing:
                if sourceTable[sourceColumn].endswith('"'):
                    result[targetColumn] = result[targetColumn] + \
                        "," + sourceTable[sourceColumn].rstrip('"')
                    capturing = False
                    capture_count += 1
                    targetColumn += 1
                else:
                    capture_count += 1
                    result[targetColumn] = result[targetColumn] + \
                        "," + sourceTable[sourceColumn]
            else:
                result.append(sourceTable[sourceColumn])
                targetColumn += 1

        return result

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")

        for c in range(len(self.column_names) - 1):
            outfile.write(str(self.column_names[c]) + ",")
        outfile.write(str(self.column_names[c + 1]) + "\n")

        for row in self.data:
            for i in range(len(row) - 1):
                outfile.write(str(row[i]) + ",")
            outfile.write(str(row[i + 1]) + "\n")

        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns:
            list of list of obj: list of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicates = []
        dont_add = []
        key_indexs = []

        for key in key_column_names:
            key_indexs.append(self.column_names.index(key))

        for i in range(len(self.data)):
            curr_row = self.data[i]
            for j in range(len(self.data)):
                count_same = 0
                for key in key_indexs:
                    if ((self.data[j][key] == curr_row[key]) and (j != i)):
                        count_same += 1
                if (count_same == len(key_indexs)):
                    count = 0
                    for x in dont_add:
                        if x == self.data[j]:
                            count += 1
                    for y in duplicates:
                        if y == self.data[j]:
                            count += 1
                    if count == 0:
                        duplicates.append(self.data[j])
                        dont_add.append(self.data[i])

        return duplicates  # TODO: fix this

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        num_rows = len(self.data)
        i = 0
        while (i < num_rows):
            for col in self.data[i]:
                if (col == "NA" or col == "N/A" or col == "Unknown" or col == "Other"):
                    self.data.pop(i)
                    i = i - 1
                    num_rows -= 1
            i += 1
    
    def remove_rows_with_na_only(self):
        """Remove rows from the table data that contain a missing value ("NA") but leave "Unknown" rows so 
        they can be reassigned.
        """
        num_rows = len(self.data)
        i = 0
        while (i < num_rows):
            for col in self.data[i]:
                if (col == "NA" or col == "N/A" ):
                    self.data.pop(i)
                    i = i - 1
                    num_rows -= 1
            i += 1

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)

        col_sum = 0
        count = 0
        for row in range(len(self.data)):
            if (self.data[row][col_index] != "NA"):
                col_sum += self.data[row][col_index]
                count += 1
        col_avg = col_sum / count

        for row in range(len(self.data)):
            for col in range(len(self.column_names)):
                if (self.data[row][col] == "NA"):
                    self.data[row][col] = col_avg

        pass  # TODO: fix this

    def replace_missing_values_with_column_average_meaningful(self, col_name, meaningful_col_name):
        """For columns with continuous data, fill missing values in a column by the column average 
        in a single meaningful_col name.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
            meaningful_col_name(str): name of column to use average of to fill missing values.
        """
        # col_index = self.column_names.index(col_name)
        # meaningful_col_index = self.column_names.index(meaningful_col_name)

        # for row in range(len(self.data)):
        #     col_sum = 0
        #     count = 0
        #     if (self.data[row][col_index] == "NA"):
        #         temp_name = self.data[row][meaningful_col_index]
        #         for r in self.data:
        #             if r[meaningful_col_index] != "NA" and r[meaningful_col_index] == temp_name:
        #                 col_sum += self.data[row][col_index]
        #                 count += 1
        #         col_avg = col_sum / count
        #         self.data[row][col_index] = col_avg

        # for row in range(len(self.data)):
        #     for col in range(len(self.column_names)):
        #         if (self.data[row][col] == "NA"):
        #             self.data[row][col] = col_avg

        pass  # TODO: fix this

    def get_trains_seperated(self, data, col_names, y_name):
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

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """

        stats_table = []
        header = ["attribute", "min", "max", "mid", "avg", "median"]

        if self.data == []:
            empt_table = MyPyTable(header, [])
            return empt_table

        col_indexs = []
        for name in col_names:
            col_indexs.append(self.column_names.index(name))

        for i in range(len(col_indexs)):
            temp_list = []
            value_list = []
            for row in self.data:
                try:
                    int(row[col_indexs[i]])
                    value_list.append(row[col_indexs[i]])
                except ValueError:
                    pass

            if len(value_list) > 0:
                list_sum = sum(value_list)
                list_avg = list_sum / len(value_list)

                list_min = min(value_list)
                list_max = max(value_list)
                list_mid = (list_max + list_min) / 2

                list_median = 0
                value_list.sort()
                if (len(value_list) % 2 == 0):
                    val = value_list[(len(value_list) // 2) - 1]
                    sec_val = value_list[len(value_list) // 2]
                    list_median = (val + sec_val) / 2
                else:
                    list_median = value_list[len(value_list) // 2]

                temp_list = [col_names[i], list_min,
                             list_max, list_mid, list_avg, list_median]
                stats_table.append(temp_list)

        new_table = MyPyTable(header, stats_table)

        return new_table  # TODO: fix this

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        header = []

        first_key_indexs = []
        second_key_indexs = []

        for name in self.column_names:
            header.append(name)
            for key in key_column_names:
                if key == name:
                    first_key_indexs.append(self.column_names.index(name))
        for name in other_table.column_names:
            count_same = 0
            for h in header:
                if h == name:
                    count_same += 1
            if count_same == 0:
                header.append(name)
            for key in key_column_names:
                if key == name:
                    second_key_indexs.append(
                        other_table.column_names.index(name))

        joined_data = copy.deepcopy(self.data)
        num_to_add = len(header) - len(self.column_names)
        for row in joined_data:
            for i in range(num_to_add):
                row.append("")

        for i in range(len(joined_data)):
            for j in range(len(other_table.data)):
                count_match = 0
                for k in range(len(first_key_indexs)):
                    if joined_data[i][first_key_indexs[k]] == other_table.data[j][second_key_indexs[k]]:
                        count_match += 1
                if count_match == len(first_key_indexs):
                    for val in range(len(other_table.data[j])):
                        col_index = header.index(other_table.column_names[val])
                        joined_data[i][col_index] = other_table.data[j][val]

        i = 0
        num_rows = len(joined_data)
        while(i < num_rows):
            for val in joined_data[i]:
                if val == "":
                    joined_data.pop(i)
                    num_rows -= 1
                    i -= 1
            i += 1

        return MyPyTable(header, joined_data)  # TODO: fix this

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        header = []

        first_key_indexs = []
        second_key_indexs = []

        for name in self.column_names:
            header.append(name)
            for key in key_column_names:
                if key == name:
                    first_key_indexs.append(self.column_names.index(name))
        for name in other_table.column_names:
            if name not in header:
                header.append(name)

        i = 0
        while(i < len(first_key_indexs)):
            for name in other_table.column_names:
                index = first_key_indexs[i]
                if name == self.column_names[index]:
                    second_key_indexs.append(
                        other_table.column_names.index(name))
                    i += 1
                    break

        # Create new joined_data starting with the self table
        # Add an extra field for each field you will need from other_table
        # Set each extra field to NA so by default something is there
        joined_data = copy.deepcopy(self.data)
        num_to_add = len(header) - len(self.column_names)
        for row in joined_data:
            for i in range(num_to_add):
                row.append("NA")

        # Run through the joined_table and look for data to fill in
        # from the other table. This will ensure it is joined left
        for i in range(len(joined_data)):
            for j in range(len(other_table.data)):
                count_match = 0
                for k in range(len(first_key_indexs)):
                    if joined_data[i][first_key_indexs[k]] == other_table.data[j][second_key_indexs[k]]:
                        count_match += 1
                if count_match == len(first_key_indexs):
                    for val in range(len(other_table.data[j])):
                        col_index = header.index(other_table.column_names[val])
                        joined_data[i][col_index] = other_table.data[j][val]

        # Run through the other_table and look for data that would not
        # have been matched with the self table. This will take care of
        # the join right.
        for j in range(len(other_table.data)):
            match_found = False
            for i in range(len(joined_data)):
                count_match = 0
                for k in range(len(second_key_indexs)):
                    if other_table.data[j][second_key_indexs[k]] == joined_data[i][first_key_indexs[k]]:
                        count_match += 1
                if count_match == len(second_key_indexs):
                    match_found = True
            if not match_found:
                new_row = []
                for x in header:
                    new_row.append("NA")
                for val in range(len(other_table.data[j])):
                    col_index = header.index(other_table.column_names[val])
                    new_row[col_index] = other_table.data[j][val]
                joined_data.append(new_row)

        return MyPyTable(header, joined_data)  # TODO: fix this

    def convert_data_to_1d_list(self):
        new_list = []
        for row in self.data:
            for col in row:
                new_list.append(col)
        return new_list
