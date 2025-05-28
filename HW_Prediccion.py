import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import pandas as pd
from  modelos_predictivos import Modelo, Prediccion

class ErrorDeCalibracion(Exception):
    """Exception raised for errors in the input."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class HW_Prediccion(Prediccion):
    def __init__(self, modelo, alpha, beta, gamma):
        super().__init__(modelo)
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

    @property
    def alpha(self):
        return self.__alpha

    @property
    def beta(self):
        return self.__beta

    @property
    def gamma(self):
        return self.__gamma

    def forecast(self, steps=1):
        res = self.modelo.forecast(steps)
        return (res)


class HW_calibrado(Modelo):
    def __init__(self, ts, test, trend='add', seasonal='add'):
        super().__init__(ts)
        self.__test = test
        self.__modelo = ExponentialSmoothing(ts, trend=trend, seasonal=seasonal, use_boxcox=True)

    @property
    def test(self):
        return self.__test

    @test.setter
    def test(self, test):
        if (isinstance(test, pd.core.series.Series)):
            if (test.index.freqstr != None):
                self.__test = test
            else:
                warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
        else:
            warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')

    def fit(self, paso=0.1, **kwargs):
        error = float("inf")
        n = np.append(np.arange(0, 1, paso), 1)
        res = None
        res_alpha = 0.2
        res_beta = 0.1
        res_gamma = 0.3
        for alpha in n:
            for beta in n:
                for gamma in n:
                    model_fit = self.__modelo.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, **kwargs)
                    pred = model_fit.forecast(len(self.test))
                    mse = sum((pred - self.test) ** 2)
                    if mse < error:
                        res_alpha = alpha
                        res_beta = beta
                        res_gamma = gamma
                        error = mse
                        res = model_fit
        if res is None:
            raise ErrorDeCalibracion("No se encontró un modelo que minimice el error.")
            return None
        return HW_Prediccion(res, res_alpha, res_beta, res_gamma)

