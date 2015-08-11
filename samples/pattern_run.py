from pattern_model import Model

SIM_STEP = 0.01
STEPS_PER_FRAME = 10

model = Model(
    neuron_count = 100
)

try :

    import imp
    imp.find_module( "matplotlib" )
    from matplotlib import pyplot as plt
    from matplotlib import animation

    figure = plt.figure()
    figure.suptitle( "Model" )

    model_plot = figure.add_subplot( 111 )
    model_plot.set_ylim( -0.5, 1.5 )
    model_plot.grid( True )

    input_line, = model_plot.plot( [], [] )
    pattern_line, = model_plot.plot( [], [] )

    time_data = []
    input_data = []
    pattern_data = []

    def animate_model( frame ) :
        for i in range( STEPS_PER_FRAME ) :
            model.step( SIM_STEP )
            time_data.append( model.time )
            input_data.append( model.input )
            pattern_data.append( model.pattern.value )
        input_line.set_data( time_data, input_data )
        pattern_line.set_data( time_data, pattern_data )
        model_plot.set_xlim( model.time - 1.0, model.time + 0.1 )
    model_animation = animation.FuncAnimation( figure, animate_model,
        interval=30 )

    plt.show()

except ImportError :

    while True :
        model.step( SIM_STEP )
