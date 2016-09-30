from collections import deque
import esn
from math import *
from matplotlib import pyplot
from matplotlib import animation

NEURON_COUNT = 256
CONNECTIVITY = 0.5
WASHOUT_TIME = 1000
TRAIN_TIME = 2000

STEP = 0.1
STEPS_PER_FRAME = 10

BETA = 2
GAMMA = 1
TAU = 2
N = 9.0

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

    print("Creating network ...")
    params = esn.NetworkParamsNSLI()
    params.inputCount = 1
    params.neuronCount = NEURON_COUNT
    params.outputCount = 1
    params.connectivity = CONNECTIVITY
    network = esn.CreateNetwork(params);

    trainerParams = esn.TrainerParams()
    trainer = esn.CreateTrainer(trainerParams, network)

    time = 0.0

    print("Washing out ...")
    while time < WASHOUT_TIME:
        network.Step(STEP)
        time = time + STEP

    print("Training ...")
    solver = MackayGlassSolver()
    while time < TRAIN_TIME:
        value = solver()
        network.Step(STEP)
        trainer.TrainOnline([value], True)
        time = time + STEP

    figure = pyplot.figure()
    subplot = figure.add_subplot(111)
    mgLine, networkLine = subplot.plot([], [], [], [])
    subplot.set_xlim(-1, 1)
    subplot.set_ylim(-1.1, 1.1)
    subplot.grid(True)

    timeData = []
    mgData = []
    networkData = []

    def animationFunc(frame):

        global time

        for i in range(0, STEPS_PER_FRAME):

            value = solver()

            network.Step(STEP)
            output = esn.Vector(1)
            network.CaptureOutput(output)

            timeData.append(time)
            mgData.append(value)
            networkData.append(output[0])
            time = time + STEP

        mgLine.set_data(timeData, mgData)
        networkLine.set_data(timeData, networkData)
        subplot.set_xlim(time - 100, time + 1)

    anim = animation.FuncAnimation(figure, animationFunc, interval=25)
    pyplot.show()
