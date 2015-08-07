from pattern_model import Model

SIM_STEP = 0.01

model = Model(
    neuron_count = 100
)

while True :
    model.step( SIM_STEP )
