import pickle  # standard python library
from mypytable import MyPyTable
from myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier
import utils as utils

# "pickle" an object (AKA object serialization)
# save a Python object to a binary file

# "unpickle" an object (AKA object de-serialization)
# load a Python object from a binary file (back into memory)

# for your project, pickle an instance MyRandomForestClassifir, MyDecisionTreeClassifier
# for demo use header and interview_tree below

pytable = MyPyTable()
pytable.load_from_file("chosen_data.csv")
print(pytable.column_names)
testing_X_train, testing_y_train, new_col_names = utils.get_trains_seperated(
    pytable.data, pytable.column_names, "stroke")

DT_classifier = MyDecisionTreeClassifier()
DT_classifier.fit(testing_X_train, testing_y_train)

# packaged_oject = [new_col_names, DT_classifier.tree]
# pickle packaged object
outfile = open("tree.p", "wb")
pickle.dump(DT_classifier, outfile)
outfile.close()
