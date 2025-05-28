import warnings
import statistics
import pandas as pd
from abc import ABCMeta, abstractmethod
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class BasePrediccion(metaclass=ABCMeta):
    @abstractmethod
    def forecast(self):
        pass


class Prediccion(BasePrediccion):
    def __init__(self, modelo):
        self.__modelo = modelo

    @property
    def modelo(self):
        return self.__modelo

    @modelo.setter
    def modelo(self, modelo):
        if (isinstance(modelo, Modelo)):
            self.__modelo = modelo
        else:
            warnings.warn('El objeto debe ser una instancia de Modelo.')


class meanfPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = []
        for i in range(steps):
            res.append(self.modelo.coef)

        start = self.modelo.ts.index[-1]
        freq = self.modelo.ts.index.freqstr
        fechas = pd.date_range(start=start, periods=steps + 1, freq=freq)
        fechas = fechas.delete(0)
        res = pd.Series(res, index=fechas)
        return (res)


class naivePrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = []
        for i in range(steps):
            res.append(self.modelo.coef)

        start = self.modelo.ts.index[-1]
        freq = self.modelo.ts.index.freqstr
        fechas = pd.date_range(start=start, periods=steps + 1, freq=freq)
        fechas = fechas.delete(0)
        res = pd.Series(res, index=fechas)
        return (res)


class snaivePrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = []
        pos = 0
        for i in range(steps):
            if pos >= len(self.modelo.coef):
                pos = 0
            res.append(self.modelo.coef[pos])
            pos = pos + 1

        start = self.modelo.ts.index[-1]
        freq = self.modelo.ts.index.freqstr
        fechas = pd.date_range(start=start, periods=steps + 1, freq=freq)
        fechas = fechas.delete(0)
        res = pd.Series(res, index=fechas)
        return (res)


class driftPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = []
        for i in range(steps):
            res.append(self.modelo.ts[-1] + self.modelo.coef * i)

        start = self.modelo.ts.index[-1]
        freq = self.modelo.ts.index.freqstr
        fechas = pd.date_range(start=start, periods=steps + 1, freq=freq)
        fechas = fechas.delete(0)
        res = pd.Series(res, index=fechas)
        return (res)


class RNN_TSPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)
        self.__scaler = MinMaxScaler(feature_range=(0, 1))
        self.__X = self.__scaler.fit_transform(self.modelo.ts.to_frame())

    def __split_sequence(self, sequence, n_steps):
        X, y = [], []
        for i in range(n_steps, len(sequence)):
            X.append(self.__X[i - n_steps:i, 0])
            y.append(self.__X[i, 0])
        return np.array(X), np.array(y)

    def forecast(self, steps=1):
        res = []
        p = self.modelo.p
        for i in range(steps):
            y_pred = [self.__X[-p:].tolist()]
            X, y = self.__split_sequence(self.__X, p)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            self.modelo.m.fit(X, y, epochs=10, batch_size=1, verbose=0)
            pred = self.modelo.m.predict(y_pred)
            res.append(self.__scaler.inverse_transform(pred).tolist()[0][0])
            self.__X = np.append(self.__X, pred.tolist(), axis=0)

        start = self.modelo.ts.index[-1]
        freq = self.modelo.ts.index.freqstr
        fechas = pd.date_range(start=start, periods=steps + 1, freq=freq)
        fechas = fechas.delete(0)
        res = pd.Series(res, index=fechas)
        return (res)


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


class ARIMA_Prediccion(Prediccion):
    def __init__(self, modelo, p, d, q):
        super().__init__(modelo)
        self.__p = p
        self.__d = d
        self.__q = q

    @property
    def p(self):
        return self.__p

    @property
    def d(self):
        return self.__d

    @property
    def q(self):
        return self.__q

    def forecast(self, steps=1):
        res = self.modelo.forecast(steps)
        return (res)


class SARIMA_Prediccion(Prediccion):
    def __init__(self, modelo, p, d, q, P, D, Q, S):
        super().__init__(modelo)
        self.__p = p
        self.__d = d
        self.__q = q
        self.__P = P
        self.__D = D
        self.__Q = Q
        self.__S = S

    @property
    def p(self):
        return self.__p

    @property
    def d(self):
        return self.__d

    @property
    def q(self):
        return self.__q

    @property
    def P(self):
        return self.__P

    @property
    def D(self):
        return self.__D

    @property
    def Q(self):
        return self.__Q

    @property
    def S(self):
        return self.__S

    def forecast(self, steps=1):
        res = self.modelo.forecast(steps)
        return (res)


class BaseModelo(metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        pass


class Modelo(BaseModelo):
    def __init__(self, ts):
        self.__ts = ts
        self._coef = None

    @property
    def ts(self):
        return self.__ts

    @ts.setter
    def ts(self, ts):
        if (isinstance(ts, pd.core.series.Series)):
            if (ts.index.freqstr != None):
                self.__ts = ts
            else:
                warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
        else:
            warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')

    @property
    def coef(self):
        return self._coef


class meanf(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self):
        self._coef = statistics.mean(self.ts)
        res = meanfPrediccion(self)
        return (res)


class naive(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self):
        self._coef = self.ts[-1]
        res = naivePrediccion(self)
        return (res)


class snaive(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self, h=1):
        self._coef = self.ts.values[-h:]
        res = snaivePrediccion(self)
        return (res)


class drift(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self):
        self._coef = (self.ts[-1] - self.ts[0]) / len(self.ts)
        res = driftPrediccion(self)
        return (res)


class RNN_TS(Modelo):
    def __init__(self, ts, p=1, lstm_units=50, dense_units=1, optimizer='rmsprop', loss='mse'):
        super().__init__(ts)
        self.__p = p
        self.__m = Sequential()
        self.__m.add(LSTM(units=lstm_units, input_shape=(p, 1)))
        self.__m.add(Dense(units=dense_units))
        self.__m.compile(optimizer=optimizer, loss=loss)

    @property
    def m(self):
        return self.__m

    @property
    def p(self):
        return self.__p

    def fit(self):
        res = RNN_TSPrediccion(self)
        return (res)


class HW_calibrado(Modelo):
    def __init__(self, ts, test, trend='add', seasonal='add'):
        super().__init__(ts)
        self.__test = test
        self.__modelo = ExponentialSmoothing(ts, trend=trend, seasonal=seasonal)

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

    def fit(self, paso=0.1):
        error = float("inf")
        alpha = paso
        while alpha <= 1:
            beta = 0
            while beta <= 1:
                gamma = 0
                while gamma <= 1:
                    model_fit = self.__modelo.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
                    pred = model_fit.forecast(len(self.test))
                    mse = sum((pred - self.test) ** 2)
                    if mse < error:
                        res_alpha = alpha
                        res_beta = beta
                        res_gamma = gamma
                        error = mse
                        res = model_fit
                    gamma += paso
                beta += paso
            alpha += paso
        return (HW_Prediccion(res, res_alpha, res_beta, res_gamma))


class ARIMA_calibrado(Modelo):
    def __init__(self, ts, test):
        super().__init__(ts)
        self.__test = test

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

    def fit(self, ar=[0, 1, 2]):
        res, res_p, res_d, res_q = (None, 0, 0, 0)
        error = float("inf")
        for p in ar:
            for d in ar:
                for q in ar:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            modelo = SARIMAX(self.ts, order=[p, d, q])
                            model_fit = modelo.fit(disp=False)
                        pred = model_fit.forecast(len(self.test))
                        mse = sum((pred - self.test) ** 2)
                        if mse < error:
                            res_p = p
                            res_d = d
                            res_q = q
                            error = mse
                            res = model_fit
                    except:
                        modelo = None
        return (ARIMA_Prediccion(res, res_p, res_d, res_q))


class SARIMA_calibrado(Modelo):
    def __init__(self, ts, test):
        super().__init__(ts)
        self.__test = test

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

    def fit(self, ar=[0, 1, 2], es=[0, 1], S=None):
        if S is None:
            warnings.warn('ERROR: No se indica el periodo a utilizar (S).')
            return (None)
        res, res_p, res_d, res_q, res_P, res_D, res_Q = (None, 0, 0, 0, 0, 0, 0)
        error = float("inf")
        for p in ar:
            for d in ar:
                for q in ar:
                    for P in es:
                        for D in es:
                            for Q in es:
                                try:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        modelo = SARIMAX(self.ts, order=[p, d, q], seasonal_order=[P, D, Q, S])
                                        model_fit = modelo.fit(disp=False)
                                    pred = model_fit.forecast(len(self.test))
                                    mse = sum((pred - self.test) ** 2)
                                    if mse < error:
                                        res_p = p
                                        res_d = d
                                        res_q = q
                                        res_P = P
                                        res_D = D
                                        res_Q = Q
                                        error = mse
                                        res = model_fit
                                except:
                                    modelo = None
        return (SARIMA_Prediccion(res, res_p, res_d, res_q, res_P, res_D, res_Q, S))