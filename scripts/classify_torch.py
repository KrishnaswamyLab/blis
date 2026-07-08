import sys 
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blis.data import traffic, cloudy, synthetic
from blis.models.GPS import GPS
import argparse
import torch
import torch_geometric.transforms as T
from blis.models.GNN_models import GCN, GAT, GIN, GNNML1, GNNML3, ChebNet, MLP, PPGN
from blis.models.blis_legs_layer import BlisNet
import argparse
import numpy as np
import tqdm
import os 
import pandas as pd

from blis.models.spectral_conv import SpectralDesign

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    if args.verbose:
        print(f'Training {args.model} on dataset {args.dataset, args.sub_dataset} on task {args.task_type}')
    total_performance = []
    #for seed in [42,43,44,45,46]:
    for seed in np.arange(1,5):

        if args.model == "GPS":
            transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        elif args.model == "ChebNet":
            transform = T.LaplacianLambdaMax(normalization="sym", is_undirected = True) # Check that all graphs are indeed undirected.
        elif args.model == "GNNML3":
            transform = SpectralDesign(nmax=0,recfield=1,dv=2,nfreq=4)
        elif args.model == "PPGN": 
            transform = SpectralDesign(nmax=-1,recfield=1,dv=2,nfreq=4)
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
            mlp_in_dim = b0.x.shape[0]
        else:
            input_dim = b0.x.shape[1]
            mlp_in_dim = b0.x.shape[0] * b0.x.shape[1]

        if args.model == "GCN":
            model = GCN(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        elif args.model == "GAT":
            model = GAT(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        elif args.model == "GIN":
            model = GIN(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        elif args.model == "GPS":
            model = GPS(in_features = input_dim, 
                        channels = args.hidden_dim, 
                        pe_dim = 8, 
                        num_layers = 2, 
                        attn_dropout = 0.5, 
                        num_classes = num_classes )
        elif args.model == "GNNML1":
            model = GNNML1(in_features = input_dim,
                        hidden_channels = args.hidden_dim,
                        num_classes = num_classes)
        elif args.model == "GNNML3":
            model = GNNML3(in_features = input_dim,
                        n_edges = train_dl.dataset[0].edge_attr2.shape[1],
                        num_classes = num_classes)
        
        elif args.model == "PPGN":
            model = PPGN(in_features = train_dl.dataset[0].X2.shape[1],
                         hidden_channels=args.hidden_dim, 
                        num_classes = num_classes)
        
        elif args.model == "ChebNet":
            model = ChebNet(in_features = input_dim, hidden_channels = args.hidden_dim, num_classes = num_classes )
        
        elif args.model == "BlisNet":
            if args.layout is not None:
                model = BlisNet(in_channels = input_dim, 
                            hidden_channels = args.hidden_dim, 
                            layout = args.layout,
                            out_channels = num_classes,
                            edge_in_channels = None,
                            trainable_laziness=False )
            else:
                model = BlisNet(in_channels = input_dim,
                                hidden_channels = args.hidden_dim,
                                out_channels = num_classes,
                                edge_in_channels = None,
                                trainable_laziness = False)
        elif args.model == 'MLP':
            #raise ValueError("Not yet implemented (at least correctly lol)")
            model = MLP(in_features = mlp_in_dim, hidden_channels = args.hidden_dim, num_classes = num_classes)
        else:
            raise ValueError("Invalid model")
        # Move the model to the specified device
        model = model.to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
                b.y = torch.tensor(b.y) # i don't recall which dataset this was necessary for
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
    #print(f"{overall_acc}, {overall_std}")
    return overall_acc, overall_std


if __name__ == "__main__":

    def csv_to_list(csv_str):
        if not csv_str:
            return []
        return csv_str.split(',')


    parser = argparse.ArgumentParser(description="Parse arguments for the program.")

    parser.add_argument("--dataset", choices=['traffic', 'partly_cloudy', 'synthetic'], help="Dataset: 'traffic' or 'partly_cloudy' or 'synthetic'.")
    parser.add_argument("--sub_dataset", help="Sub-dataset value depending on the dataset chosen.")
    parser.add_argument("--task_type", type=str,  help="The task type to use for the classification")
    parser.add_argument("--model", nargs = '+', type=str, default="GCN", help="Model: GCN, GAT, GIN, ChebNet, BlisNet, GNNML1, GNNML3, MLP")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Number of hidden channels in the GNN model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type = float, default = .001, help="Optimizer learning rate")
    parser.add_argument("--verbose", type=int, default=1, help="Print training, either 0 or 1")
    parser.add_argument('--layout', type=csv_to_list, help='Layout to use for the BliNet experiments e.g. blis,blis,dim_reduction,gcn')


    args = parser.parse_args()

    # generate the list of sub datasets 
    if args.dataset == 'traffic' and args.sub_dataset == 'full':
        sub_datasets = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
    elif args.dataset == 'partly_cloudy' and args.sub_dataset == 'full':
        sub_datasets = [f'{i:04d}' for i in range(155)]
    elif args.dataset == 'synthetic' and args.sub_dataset == 'full':
        dataset_types = ['camel_pm', 'gaussian_pm']
        sub_datasets = []
        for dataset_type in dataset_types:
            for i in range(5):
                sub_datasets.append(f'{dataset_type}_{i}')
    else:
        sub_datasets = [args.sub_dataset]

    results_list = []
    model_list = args.model 
    if args.model[0] == 'all':
        model_list = ['GCN', 'GAT', 'GIN', 'ChebNet', 'BlisNet', 'GNNML1', 'GNNML3']

    for model in model_list:
        for sub_dataset in sub_datasets:
            args.sub_dataset = sub_dataset 
            args.model = model 
            score, stdev = main(args)

            new_row = {
                'model': args.model,
                'hidden_dim': args.hidden_dim,
                'epochs': args.epochs, 
                'learning_rate': args.learning_rate,
                'task_type': args.task_type, 
                'dataset': args.dataset, 
                'sub_dataset': sub_dataset,
                'score': score, 
                'stdev': stdev
            }
            results_list.append(new_row)

    df_results = pd.DataFrame(results_list)
    if len(sub_datasets) > 1:
        sub_dataset = 'full'
    if len(model_list) > 1:
        args.model = 'multi-model'

    save_name = f'10fold_{args.dataset}_{sub_dataset}_{args.model}_{args.hidden_dim}_{args.task_type}.csv'
    print(save_name)
    save_dir='run_results'
    if os.path.exists(save_dir):
        df_results.to_csv(os.path.join(save_dir,save_name), index = False)
    else:
        os.mkdir(save_dir)
        df_results.to_csv(os.path.join(save_dir,save_name), index = False)

    #main(args)


    #Example : python classify_torch.py --dataset partly_cloudy --sub_dataset 0001 --task_type EMOTION3 --model BlisNet


