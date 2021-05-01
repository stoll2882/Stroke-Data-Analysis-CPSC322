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


def bar_chart(x, y, title):
    '''
    This function creates a standard bar chart

    Attributes:
        x: list of x axis labels
        y: list of corresponding y values 
        title: title for the chart
    '''
    plt.figure()
    plt.bar(x, y)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.title(title)
    plt.show()