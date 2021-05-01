import matplotlib.pyplot as plt

def pie_chart(x, y, title):
    '''
    This function creates a standard pie chart
    Attributes:
        x: list of labels
        y: list of values
        title: title for the chart
    '''
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.title(title)
    plt.show()