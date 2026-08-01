from domain.train_ltwr import TrainLTWR
from domain.run_experiment import RunExperiment
from domain.visiualize_data import VisualizeData

def main():
    # Train the LTWR model
    TrainLTWR()

    # Run the experiment using the trained model
    RunExperiment()

    # Visualize the results of the experiment
    VisualizeData()



if __name__ == "__main__":
    main()