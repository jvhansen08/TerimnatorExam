from ml_model import displayAgent
from controller import evaluatePIDController
from cartpole import noTraining


def main():
    print("\n---- Demoing the model without training or a controller ----\n")
    noTraining()
    print("\n---- Demoing the model with a PID controller ----\n")
    evaluatePIDController(human=True)
    print("\n---- Demoing the model with training ----\n")
    displayAgent()


if __name__ == "__main__":
    main()
