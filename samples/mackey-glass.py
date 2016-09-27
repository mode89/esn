from collections import deque
import esn
from math import *
from matplotlib import pyplot
from matplotlib import animation

STEP = 0.1
STEPS_PER_FRAME = 100
BETA = 2
GAMMA = 1
TAU = 2
N = 9.65

class MackayGlassSolver:

    def __init__(self):
        length = int(TAU/STEP) + 1
        self.x = deque([0.9] * length, maxlen=length)

    def __call__(self):
        xtau = self.x.popleft()
        xt0 = self.x[-1]
        dx = (BETA * xtau / (1.0 + pow(xtau, N)) - GAMMA * xt0) / \
            (GAMMA + 1 / STEP)
        xt1 = xt0 + dx
        self.x.append(xt1)
        return xt1 - 1.0

if __name__ == "__main__":
    figure = pyplot.figure()
    subplot = figure.add_subplot(111)
    mgLine, = subplot.plot([], [])
    subplot.set_xlim(-1, 1)
    subplot.set_ylim(-1.1, 1.1)
    subplot.grid(True)

    timeData = []
    mgData = []
    outputData = []

    time = 0.0
    solver = MackayGlassSolver()

    def animationFunc(frame):
        global time
        for i in range(0, STEPS_PER_FRAME):
            timeData.append(time)
            mgData.append(solver())
            time = time + STEP
        mgLine.set_data(timeData, mgData)
        subplot.set_xlim(time - 100, time + 1)

    anim = animation.FuncAnimation(figure, animationFunc, interval=25)
    pyplot.show()
