from blis.data import traffic, cloudy, synthetic
from blis.models.GPS import GPS
import argparse
import torch
import torch_geometric.transforms as T

from blis.models.GNN_models import GCN, GAT, GIN
import argparse
import numpy as np
import tqdm

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    total_performance = []
    for seed in [42,43,44,45,56]:

        if args.model == "GPS":
            transform = transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        else:
            transform = None

        if args.dataset == "traffic":
            train_dl, test_dl, num_classes = traffic.traffic_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32, transform = transform)
        elif args.dataset == "partly_cloudy":
            train_dl, test_dl, num_classes = cloudy.cloudy_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32, transform = transform)
        elif args.dataset == "synthetic":
            train_dl, test_dl, num_classes = synthetic.synthetic_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32, transform = transform)
        else:
            raise ValueError("Invalid dataset")

        b0 = next(iter(train_dl))
        if len(b0.x.shape) == 1:
            input_dim = 1
        else:
            input_dim = b0.x.shape[1]

        if args.model == "GCN":
            model = GCN(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        elif args.model == "GAT":
            model = GCN(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        elif args.model == "GIN":
            model = GIN(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        elif args.model == "GPS":
            model = GPS(in_features = input_dim, 
                        channels = args.hidden_dim, 
                        pe_dim = 8, 
                        num_layers = 2, 
                        attn_dropout = 0.5, 
                        num_classes = num_classes )
        else:
            raise ValueError("Invalid model")
        
        # Move the model to the specified device
        model = model.to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # Define the loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in tqdm.tqdm(range(args.epochs)):
            model.train()
            total_loss = 0

            for i, b in enumerate(train_dl):
                # Move data to the specified device
                b = b.to(device)

                optimizer.zero_grad()

                # Forward pass
                out = model(b)
                loss = criterion(out, b.y.to(device))

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            if epoch%10 ==0 and args.verbose:
                print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / len(train_dl)}")

        # Testing loop
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for b in test_dl:
                # Move data to the specified device
                b = b.to(device)

                out = model(b)
                _, predicted = torch.max(out, 1)
                b.y = torch.tensor(b.y)
                total += b.y.size(0)
                #import pdb; pdb.set_trace()
                correct += (predicted.detach().cpu() == b.y.cpu()).sum().item()
        if args.verbose:
            print(f"Accuracy on test set: {100 * correct / total}%")
        total_performance.append(100 * correct / total)
    overall_acc = np.mean(np.array(total_performance))
    overall_std = np.std(np.array(total_performance))
    if args.verbose:
        print(f"Mean overall performance is {overall_acc}, standard dev is {overall_std}")
    print(f"{overall_acc}, {overall_std}")


            

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments for the program.")

    parser.add_argument("--dataset", choices=['traffic', 'partly_cloudy', 'synthetic'], help="Dataset: 'traffic' or 'partly_cloudy' or 'synthetic'.")
    parser.add_argument("--sub_dataset", help="Sub-dataset value depending on the dataset chosen.")
    parser.add_argument("--task_type", type=str,  help="The task type to use for the classification")
    parser.add_argument("--model", type=str, default="GCN", help="Classification model to use")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Number of hidden channels in the GNN model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type = float, default = .001, help="Optimizer learning rate")
    parser.add_argument("--verbose", type=int, default=0, help="Print training, either 0 or 1")

    args = parser.parse_args()

    main(args)

    #Example : python classify_torch.py --dataset partly_cloudy --sub_dataset 0001 --task_type EMOTION3


