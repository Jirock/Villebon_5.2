import numpy as np

def msle(y, y_pred):
    pass

def main():
    y_true = np.array([3, 5, 2.5, 7])
    y_pred = np.array([2.5, 5, 4, 8])
    np.testing.assert_almost_equal(msle(y_true, y_pred), 0.039, decimal=3)


if __name__ == '__main__':
    main()