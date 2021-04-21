import copy
import csv 

#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
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


    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            1d list of items in col

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = self.column_names.index(col_identifier)
        col = []

        for row in len(self):
            if (row[col_index] != "NA"):
                col.append(row[col_index])
        return col  

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        #in class version of convert_to_numeric
        for value in range(len(self.data)): 
            try:
                numeric_value = float(self.data[value])
                self.data[value] = numeric_value
            except ValueError:
                print(self.data[value], "cant be converted")
             
       

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """

        #find rows to drop in self.data
        #MyPyTable object -> self.data
        cols = self.column_names
        for row in len(cols):
            for i in rows_to_drop:
                if self.column_names[row] == rows_to_drop[i]:
                    #col = get_column(self, header[i], True)
                    self.column_names.remove(row)
                    #self.data??


    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        the_file = open(filename,'r')
        the_reader = csv.reader(the_file)
        table = []
        for row in the_reader:
            if len(row)> 0:
                table.append(row)
        
        the_file.close
        return table

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        col_names = self.column_names
        out_file = open(filename, 'w')
        file_writer = csv.writer(out_file)
    
        file_writer.writerows(self.data)
        file_writer.writerow(col_names)
        out_file.close
    
    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        #go through table row by row
        for row in len(self):
            while("NA" in row): #if?
                self.data.remove(row)
         

    
