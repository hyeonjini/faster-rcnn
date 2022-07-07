import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -- Hyperparameters
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--use_drop", type=bool, default=False)
    parser.add_argument("--rpn_sigma", type=float, default=3.0)
    parser.add_argument("--roi_sigma", type=float, default=1.0)

    # -- Path
    parser.add_argument("--data_dir", type=str, default="")
    
    
