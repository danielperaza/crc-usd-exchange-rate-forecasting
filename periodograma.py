import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from plotly import graph_objects as go

class Periodograma:
    def __init__(self, ts):
        self.__ts = ts
        self.__freq, self.__spec = signal.periodogram(ts)

    @property
    def ts(self):
        return self.__ts

    @property
    def freq(self):
        return self.__freq

    @property
    def spec(self):
        return self.__spec

    def mejor_freq(self, best=3):
        res = np.argsort(-self.spec)
        res = res[res != 1][0:best]
        return (self.freq[res])

    def mejor_periodos(self, best=3):
        return (1 / self.mejor_freq(best))

    def plot_periodograma(self, best=3):
        res = self.mejor_freq(best)
        plt.plot(self.freq, self.spec, color="darkgray")
        for i in range(best):
            plt.axvline(x=res[i], label="Mejor " + str(i + 1), ls='--', c=np.random.rand(3))
        plt.legend(loc="best")
        plt.show()

    def plotly_periodograma(self, best=3):
        res = self.mejor_freq(best)
        fig = go.Figure()
        no_plot = fig.add_trace(
            go.Scatter(x=self.freq, y=self.spec,
                       mode='lines+markers', line_color='darkgray')
        )
        for i in range(best):
            v = np.random.rand(3)
            color = "rgb(" + str(v[0]) + ", " + str(v[1]) + ", " + str(v[2]) + ")"
            fig.add_vline(x=res[i], line_width=2, line_dash="dash",
                          annotation_text="Mejor " + str(i + 1),
                          line_color=color)

        return (fig)