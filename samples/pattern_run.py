from pattern_model import PatternModel

SIM_STEP = 0.01

model = PatternModel(
    neuron_count = 100
)

while True :
    model.step( SIM_STEP )
