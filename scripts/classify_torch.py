from blis.data import traffic, cloudy, synthetic
from blis.models.GCN import GCN
import argparse


def main(args):

    for seed in [42,43,44,45,56]:

        if args.dataset == "traffic":
            train_dl, val_dl, test_dl, num_classes = traffic.traffic_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32)
        elif args.dataset == "partly_cloudy":
            train_dl, val_dl, test_dl, num_classes = cloudy.cloudy_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32)
        elif args.dataset == "synthetic":
            train_dl, val_dl, test_dl, num_classes = synthetic.synthetic_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32)
        else:
            raise ValueError("Invalid dataset")


        b0 = next(iter(train_dl))
        input_dim = b0.x.shape[1]

        if args.model == "GCN":
            model = GCN(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        

        for epoch in range(args.epochs):
            ### TRAINING LOOP
            break 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments for the program.")

    parser.add_argument("--dataset", choices=['traffic', 'partly_cloudy', 'synthetic'], help="Dataset: 'traffic' or 'partly_cloudy' or 'synthetic'.")
    parser.add_argument("--sub_dataset", help="Sub-dataset value depending on the dataset chosen.")
    parser.add_argument("--task_type", type=str,  help="The task type to use for the classification")
    parser.add_argument("--model", type=str, default="LR", help="Classification model to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")

    args = parser.parse_args()

    main(args)

    #Example : python classify_scattering.py --dataset=traffic --largest_scale=4 --sub_dataset=PEMS04 --scattering_type=blis --task_type=DAY


