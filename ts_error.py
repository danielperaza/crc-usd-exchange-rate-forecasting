import math
import warnings

from numpy import corrcoef
import numpy
import pandas as pd
from plotly import graph_objects as go
from matplotlib import pyplot as plt

class ts_error:
    def __init__(self, preds, real, nombres=None):
        self.__preds = preds
        self.__real = real
        self.__nombres = nombres

    @property
    def preds(self):
        return self.__preds

    @preds.setter
    def preds(self, preds):
        if (isinstance(preds, pd.core.series.Series) or isinstance(preds, numpy.ndarray)):
            self.__preds = [preds]
        elif (isinstance(preds, list)):
            self.__preds = preds
        else:
            warnings.warn('ERROR: El parámetro preds debe ser una serie de tiempo o una lista de series de tiempo.')

    @property
    def real(self):
        return self.__real

    @real.setter
    def real(self, real):
        self.__real = real

    @property
    def nombres(self):
        return self.__nombres

    @nombres.setter
    def nombres(self, nombres):
        if (isinstance(nombres, str)):
            nombres = [nombres]
        if (len(nombres) == len(self.__preds)):
            self.__nombres = nombres
        else:
            warnings.warn('ERROR: Los nombres no calzan con la cantidad de métodos.')

    def RSS(self):
        res = []
        for pred in self.preds:
            res.append(sum((pred - self.real) ** 2))
        return (res)

    def MSE(self):
        return ([pred / len(self.real) for pred in self.RSS()])

    def RMSE(self):
        return ([math.sqrt(pred) for pred in self.MSE()])

    def RE(self):
        res = []
        for pred in self.preds:
            res.append(sum(abs(self.real - pred)) / sum(abs(self.real)))
        return (res)

    def CORR(self):
        res = []
        for pred in self.preds:
            try:
                # Verificar que los datos no sean constantes (evita división por cero)
                if numpy.std(self.real) == 0 or numpy.std(pred) == 0:
                    res.append(0)  # No hay correlación si alguna serie es constante
                else:
                    # Usar try-except para capturar otros errores numéricos
                    corr = corrcoef(self.real, pred)[0, 1]
                    res.append(0 if math.isnan(corr) else corr)
            except Exception as e:
                print(f"Error al calcular correlación: {e}")
                res.append(0)  # En caso de error, asignar correlación cero
        return res

    def df_errores(self):
        res = pd.DataFrame({'MSE': self.MSE(), 'RMSE': self.RMSE(), 'RE': self.RE(), 'CORR': self.CORR()})
        if (self.nombres is not None):
            res.index = self.nombres
        return (res)

    def __escalar(self):
        res = self.df_errores()
        for nombre in res.columns.values:
            res[nombre] = res[nombre] - min(res[nombre])
            res[nombre] = res[nombre] / max(res[nombre]) * 100
        return (res)

    def plot_errores(self):
        plt.figure(figsize=(8, 8))
        df = self.__escalar()
        if (len(df) == 1):
            df.loc[0] = 100

        N = len(df.columns.values)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)

        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], df.columns.values)

        ax.set_rlabel_position(0)
        plt.yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], color="grey", size=10)
        plt.ylim(-10, 110)

        for i in df.index.values:
            p = df.loc[i].values.tolist()
            p = p + p[:1]
            ax.plot(angles, p, linewidth=1, linestyle='solid', label=i)
            ax.fill(angles, p, alpha=0.1)

        plt.legend(loc='best')
        plt.show()

    def plotly_errores(self):
        df = self.__escalar()
        etqs = df.columns.values.tolist()
        etqs = etqs + etqs[:1]
        if (len(df) == 1):
            df.loc[0] = 100

        fig = go.Figure()

        for i in df.index.values:
            p = df.loc[i].values.tolist()
            p = p + p[:1]
            fig.add_trace(go.Scatterpolar(
                r=p, theta=etqs, fill='toself', name=i
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-10, 110]
                ))
        )

        return (fig)