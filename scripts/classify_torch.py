from blis.data import traffic, cloudy, synthetic
from blis.models.GNN_models import GCN, GAT, GIN
import argparse
import torch


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        import pdb; pdb.set_trace()
        b0 = next(iter(train_dl))
        input_dim = int(b0.x.shape[0]/len(b0.y))

        if args.model == "GCN":
            model = GCN(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        if args.model == "GAT":
            model = GCN(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        if args.model == "GIN":
            model = GIN(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )

        # Move the model to the specified device
        model = model.to(device)

        # Define the optimizer
        learning_rate = .001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Define the loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0

            for i, b in enumerate(train_dl):
                # Move data to the specified device
                b = b.to(device)

                optimizer.zero_grad()

                # Forward pass
                out = model(b.x, b.edge_index, b.batch)
                loss = criterion(out, b.y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / len(train_dl)}")

        # Testing loop
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for b in test_dl:
                # Move data to the specified device
                b = b.to(device)

                out = model(b.x, b.edge_index, b.batch)
                _, predicted = torch.max(out, 1)
                total += b.y.size(0)
                correct += (predicted == b.y).sum().item()

        print(f"Accuracy on test set: {100 * correct / total}%")

            

            

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

    #Example : python classify_torch.py --dataset partly_cloudy --sub_dataset 0001 --task_type EMOTION3


