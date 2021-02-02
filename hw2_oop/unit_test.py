import warnings
import numpy as np
import unittest
from numpy.testing import assert_allclose

class LogisticRegressionGD:
    '''
    A simple logistic regression for binary classification with gradient descent
    '''
    
    def __init__(self):
        pass
    
    def sigmoid(self, X, W):
        return 1/(1+np.exp(-(X @ W)))
    
    def __extend_X(self, X):
        """
            Данный метод должен возвращать следующую матрицу:
            X_ext = [1, X], где 1 - единичный вектор
            это необходимо для того, чтобы было удобнее производить
            вычисления, т.е., вместо того, чтобы считать X@W + b
            можно было считать X_ext@W_ext 
        """
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
   
    def init_weights(self, input_size, output_size):
        """
            Инициализирует параметры модели
            W - матрица размерности (input_size, output_size)
            инициализируется рандомными числами из
            нормального распределения со средним 0 и стандартным отклонением 0.01
        """
        np.random.seed(42)
        self.W = np.random.normal(loc=0, scale=0.01, size=(input_size, output_size))
        
    def get_loss(self, p, y):
        """
            Данный метод вычисляет логистическую функцию потерь
            @param p: Вероятности принадлежности к классу 1
            @param y: Истинные метки
        """
        return -np.sum(y*np.log(p) + (1-y)*np.log(1-p))/y.shape[0]
    
    def get_prob(self, X):
        """
            Данный метод вычисляет P(y=1|X,W)
            Возможно, будет удобнее реализовать дополнительный
            метод для вычисления сигмоиды
        """
        if X.shape[1] != self.W.shape[0]:
            X = self.__extend_X(X)
        return self.sigmoid(X, self.W)
    
    def get_acc(self, p, y, threshold=0.5):
        """
            Данный метод вычисляет accuracy:
            acc = \frac{1}{len(y)}\sum_{i=1}^{len(y)}{I[y_i == (p_i >= threshold)]}
        """
        sum = 0
        for i in range(len(y)):
            if y[i]==(p[i]>=threshold):
                sum += 1
        return sum/len(y)

    def fit(self, X, y, num_epochs=100, lr=0.001):
        
        X = self.__extend_X(X)
        self.init_weights(X.shape[1], y.shape[1])
        
        accs = []
        losses = []
        for _ in range(num_epochs):
            p = self.get_prob(X)

            W_grad = (X.T @ (p-y))/y.shape[0]
            self.W -= W_grad
            
            # необходимо для стабильности вычислений под логарифмом
            p = np.clip(p, 1e-10, 1 - 1e-10)
            
            log_loss = self.get_loss(p, y)
            losses.append(log_loss)
            acc = self.get_acc(p, y)
            accs.append(acc)
        
        return accs, losses

class LogisticRegressionGDTest(unittest.TestCase):

    def test_get_loss(self):
        linear_regression = LogisticRegressionGD()
        p = np.array([0.8, 0.9, 0.1]).reshape(-1, 1)
        y = np.array([1, 0, 1]).reshape(-1, 1)

        loss = linear_regression.get_loss(p, y)
        loss_true = -(np.log(0.8) + np.log(0.1) + np.log(1 - 0.9)) / 3

        self.assertAlmostEqual(loss, loss_true)

    def test_get_prob(self):
        linear_regression = LogisticRegressionGD()
        linear_regression.W = np.random.rand(3, 1)
        X = np.ones((4, 3))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob_true = 1 / (1 + np.exp(-X @ linear_regression.W))
        prob = linear_regression.get_prob(X)

        assert_allclose(prob, prob_true)

    def test_get_acc(self):
        linear_regression = LogisticRegressionGD()

        p = np.array([0.8, 0.9, 0.1, 0.0, 1.0]).reshape(-1, 1)
        y = np.array([1, 0, 1, 1, 1]).reshape(-1, 1)
        acc_true = 2 / 5
        acc = linear_regression.get_acc(p, y, threshold=0.5)

        self.assertAlmostEqual(acc, acc_true)

if __name__ == '__main__':
    unittest.main()