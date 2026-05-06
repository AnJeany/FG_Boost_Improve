from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from autofeat import AutoFeatClassifier

from src.dimRedu.randomFourier import RFFGaussianProcess
from src.train.hpca import optht

'''
- transformations: list of transformations that should be applied; possible elements:
        "1/", "exp", "log", "abs", "sqrt", "^2", "^3", "1+", "1-", "sin", "cos", "exp-", "2^"
      (first 7, i.e., up to ^3, are applied by default)
'''

class transformData:
    def __init__(self, X, y, x_val, x_test):
        self.X = X
        self.y = y
        self.x_val = x_val
        self.x_test = x_test
    def _transform_robuster(self):
        scaler = RobustScaler()
        x_new = scaler.fit_transform(self.X)
        x_test_new =  scaler.transform(self.x_test)
        x_val_new = scaler.transform(self.x_val)
        return x_new, x_test_new, x_val_new
    def _transform_minmax(self):
        scaler = MinMaxScaler()
        x_new = scaler.fit_transform(self.X)
        x_test_new =  scaler.transform(self.x_test)
        x_val_new = scaler.transform(self.x_val)
        return x_new, x_test_new, x_val_new
    def _transform_hpca(self):
        train_x_torch = np.array(self.X)
        val_x_torch = np.array(self.x_val)
        test_x_torch = np.array(self.x_test)
        # Perform SVD
        u, s, vh = np.linalg.svd(train_x_torch, full_matrices=False)
        v = vh.T
        k = optht(train_x_torch, sv=s, sigma=None)
        # Project train and validation data by V
        if k == 0 or k == 1:
            x_new = np.dot(train_x_torch, v)
            x_test_new = np.dot(test_x_torch, v)
            x_val_new = np.dot(val_x_torch, v)
        else:
            x_new = np.dot(train_x_torch, v[:, :k])
            x_test_new =  np.dot(test_x_torch, v[:, :k])
            x_val_new = np.dot(val_x_torch, v[:, :k])
        return x_new, x_test_new, x_val_new
    def _transform_pcaCosSin(self):
        train_x_torch = np.array(self.X)
        val_x_torch = np.array(self.x_val)
        test_x_torch = np.array(self.x_test)
        # Perform SVD
        u, s, vh = np.linalg.svd(train_x_torch, full_matrices=False)
        v = vh.T
        k = optht(train_x_torch, sv=s, sigma=None)

        # Perform SVD
        # u, s, vh = np.linalg.svd(self.X, full_matrices=False)
        # v = vh.T
        # k = optht(self.X, sv=s, sigma=None)
        # # Project train and validation data by V
        # x_new = np.dot(self.X, v[:, :k])
        # x_test_new =  np.dot(self.x_test, v[:, :k])
        # x_val_new = np.dot(self.x_val, v[:, :k])
        # return x_new, x_test_new, x_val_new
    def _transform_autofeat(self):
        units = {}
        afreg = AutoFeatClassifier(verbose=1, feateng_steps=1, units=units)
        x_new = afreg.fit_transform(self.X, self.y)
        x_test_new = afreg.transform(self.x_test)
        x_val_new = afreg.transform(self.x_val)
        return x_new, x_test_new, x_val_new

    def _transform_randomProject(self):
        randF = RFFGaussianProcess()
        x_new = randF._get_rffs(self.X)
        x_test_new = randF._get_rffs(self.x_test)
        x_val_new = randF._get_rffs(self.x_val)
        return x_new.T, x_test_new.T, x_val_new.T

        # pca-cos-sin,
        # # Logarithmic Transformation
        # df['Age_log'] = np.log(df['Age'])
        # # Reciprocal Trnasformation
        # df['Age_reciprocal'] = 1 / df.Age
        # # Square Root Transformation
        # df['Age_sqaure'] = df.Age ** (1 / 2)
        # # Exponential Transdormation
        # df['Age_exponential'] = df.Age ** (1 / 1.2)
        # BoxCOx Transformation
        # df['Age_Boxcox'], parameters = stat.boxcox(df['Age'])
        # # Fare
        # df['Fare_log'] = np.log1p(df['Fare'])