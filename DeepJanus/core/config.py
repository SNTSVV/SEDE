class Config:
    GEN_RANDOM = 'GEN_RANDOM'
    GEN_RANDOM_SEEDED = 'GEN_RANDOM_SEEDED'
    GEN_SEQUENTIAL_SEEDED = 'GEN_SEQUENTIAL_SEEDED'

    SEG_LENGTH = 25
    NUM_SPLINE_NODES =10
    INITIAL_NODE = (0.0, 0.0, -28.0, 8.0)
    ROAD_BBOX_SIZE = (-250, 0, 250, 500)

    def __init__(self):
        self.experiment_name = 'exp'
        self.fitness_weights = (1.0, -1.0)
        #?

        #self.POPSIZE = 12
        #self.POPSIZE = 24
        self.POPSIZE = 4
        #change to 25
        #self.NUM_GENERATIONS = 150
        #self.NUM_GENERATIONS = 1
        self.GEN_MINUTES = 2400
        #self.GEN_MINUTES = 60
        #change to num_generations of SEDE

        self.RESEED_UPPER_BOUND = int(self.POPSIZE * 0.1)

        self.MUTATION_EXTENT = 6.0
        #self.ARCHIVE_THRESHOLD = 35.0
        self.ARCHIVE_THRESHOLD = 0.0
        #change to 0
        self.duplicateCount = 0

        self.K_SD = 0.01

        self.simulation_save = True
        self.simulation_name = 'beamng_nvidia_runner/sim_$(id)'

        #self.keras_model_file = 'self-driving-car-4600.h5'
        self.keras_model_file = 'self-driving-car-185-2020.h5'
        # add HPD_model_file
        # add FLD_model_file

        self.generator_name = Config.GEN_RANDOM
        #self.generator_name = Config.GEN_RANDOM_SEEDED
        #self.generator_name = Config.GEN_SEQUENTIAL_SEEDED
        # change to GEN_RANDOM

        self.seed_folder = 'population_HQ1'
