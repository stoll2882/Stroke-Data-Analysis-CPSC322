from MyPyTable import MyPyTable
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def main():
    stroke_table = MyPyTable()
    stroke_table.load_from_file("healthcare-dataset-stroke-data.csv")
    stroke_table.remove_rows_with_missing_values()
    stroke_table.convert_to_numeric()
    print(stroke_table.get_shape())
    # Classify all string types into numerical values
    stroke_table.classify_column("gender")
    stroke_table.classify_column("ever_married")
    stroke_table.classify_column("work_type")
    stroke_table.classify_column("Residence_type")
    stroke_table.classify_column("smoking_status")
    # seperate into x and y trains
    X, y, col_names = stroke_table.get_trains_seperated(
        stroke_table.data, stroke_table.column_names, "stroke")
    # select the 6 best attributes to split on
    X_new = SelectKBest(k=6).fit_transform(X, y)
    print(X_new)
    print(len(X_new))


if __name__ == "__main__":
    main()
