import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def question_1(data_file):
    data = np.loadtxt("data_cc6.csv")
    x, y = data[:,0], data[:,1]
    return x,y

def question_2(x, y):
    plt.hist(x)
    plt.title('histogramme x')
    plt.figure()
    plt.hist(y)
    plt.title('histogramme y')
    plt.figure()
    plt.title('scatter x, y')
    plt.scatter(x,y)
    plt.show()

def question_3(x, y):
    regr = LinearRegression()
    x = x.reshape(-1,1)
    regr.fit(x,y)
    residus = y - regr.predict(x)
    plt.hist(residus)
    plt.title('histogramme residus')
    plt.show()

def main():
    x, y = question_1("data_cc6.csv")
    question_2(x,y)
    question_3(x,y)


if __name__ == '__main__':
    main()