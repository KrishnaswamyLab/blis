from blis.data import traffic, cloudy, synthetic
from blis.models.GCN import GCN
from blis.models.GPS import GPS
import argparse
import torch
import torch_geometric.transforms as T



def main(args):

    for seed in [42,43,44,45,56]:

        if args.model == "GPS":
            transform = transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        else:
            transform = None

        if args.dataset == "traffic":
            train_dl, val_dl, test_dl, num_classes = traffic.traffic_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32, transform = transform)
        elif args.dataset == "partly_cloudy":
            train_dl, val_dl, test_dl, num_classes = cloudy.cloudy_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32, transform = transform)
        elif args.dataset == "synthetic":
            train_dl, val_dl, test_dl, num_classes = synthetic.synthetic_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32, transform = transform)
        else:
            raise ValueError("Invalid dataset")


        b0 = next(iter(train_dl))
        input_dim = b0.x.shape[1]

        if args.model == "GCN":
            model = GCN(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        elif args.model == "GPS":
            # attn type can be "multihead" or "performer"
            model = GPS(in_features = input_dim, 
                        channels = args.hidden_dim, 
                        pe_dim = 8, 
                        num_layers = 2, 
                        attn_type = "multihead", 
                        attn_kwargs = {'dropout': 0.5}, 
                        num_classes = num_classes )
        
        for epoch in range(args.epochs):

            for i,b in enumerate(train_dl):
                model.zero_grad()
                preds = model(b)
                loss = 0
                #COMPLETE LOSS FUNCTION
            

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments for the program.")

    parser.add_argument("--dataset", choices=['traffic', 'partly_cloudy', 'synthetic'], help="Dataset: 'traffic' or 'partly_cloudy' or 'synthetic'.")
    parser.add_argument("--sub_dataset", help="Sub-dataset value depending on the dataset chosen.")
    parser.add_argument("--task_type", type=str,  help="The task type to use for the classification")
    parser.add_argument("--model", type=str, default="GCN", help="Classification model to use")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Number of hidden channels in the GNN model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")

    args = parser.parse_args()

    main(args)

    #Example : python classify_scattering.py --dataset=traffic --largest_scale=4 --sub_dataset=PEMS04 --scattering_type=blis --task_type=DAY


