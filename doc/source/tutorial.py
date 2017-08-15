# Based on: http://isadoranun.github.io/tsfeat/FeaturesDocumentation.html

# JSAnimation import available at https://github.com/jakevdp/JSAnimation

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation

from JSAnimation import IPython_display

from IPython.display import Image, HTML, YouTubeVideo

# introduction
def ts_anim():
    # create a simple animation
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 100), ylim=(-1, 1))
    Color = [ 1 ,0.498039, 0.313725];
    line, = ax.plot([], [], '*',color = Color)
    plt.xlabel("Time")
    plt.ylabel("Measurement")

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = np.linspace(0, i+1, i+1)
        ts = 5*np.cos(x * 0.02 * np.pi) * np.sin(np.cos(x)  * 0.02 * np.pi)
        line.set_data(x, ts)
        return line,

    return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=200, blit=True)


def macho_video():
    return YouTubeVideo('qMx4ozpSRuE',  width=750, height=360, align='right')


def macho_example11():
    picture = Image(filename='_static/curvas_ejemplos11.jpg')
    picture.size = (100, 100)
    return picture

# the library

magnitude_ex = np.random.rand(30)
time_ex = np.arange(0, 30)