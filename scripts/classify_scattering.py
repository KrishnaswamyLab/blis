from blis.data import traffic, cloudy, synthetic
import argparse
from sklearn.linear_model import LogisticRegression


def main(args,scattering_dict):

    for seed in [42,43,44,45,56]:

        if args.dataset == "traffic":
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = traffic.traffic_scattering_data_loader(seed=42,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32,
                                                                                                    scattering_dict=scattering_dict)
        elif args.dataset == "partly_cloudy":
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = cloudy.cloudy_scattering_data_loader(seed=42,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32,
                                                                                                    scattering_dict=scattering_dict)
        elif args.dataset == "synthetic":
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = synthetic.synthetic_scattering_data_loader(seed=42,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    batch_size=32,
                                                                                                    scattering_dict=scattering_dict)
        else:
            raise ValueError("Invalid dataset")



        X_train = X_train.reshape(X_train.shape[0],-1)
        X_val = X_val.reshape(X_val.shape[0],-1)
        X_test = X_test.reshape(X_test.shape[0],-1)

        if args.model == "LR":
            model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
            print("Train score : ", model.score(X_train, y_train))
            print("Val score : ", model.score(X_val, y_val))
            print("Test score : ", model.score(X_test, y_test))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments for the program.")

    parser.add_argument("--scattering_type", choices=['blis', 'modulus'], help="Type of scattering: 'blis' or 'modulus'.")
    parser.add_argument("--largest_scale", type=int, help="Largest (dyadic) scale as a positive integer.")
    parser.add_argument("--highest_moment", type=int, default=1, help="Highest moment as a positive integer. Defaults to 1.")
    parser.add_argument("--dataset", choices=['traffic', 'partly_cloudy', 'synthetic'], help="Dataset: 'traffic' or 'partly_cloudy' or 'synthetic'.")
    parser.add_argument("--sub_dataset", help="Sub-dataset value depending on the dataset chosen.")
    parser.add_argument("--num_layers", type=int, default=2, help="Largest scattering layer")
    parser.add_argument("--task_type", type=str,  help="The task type to use for the classification")
    parser.add_argument("--model", type=str, default="LR", help="Classification model to use")

    args = parser.parse_args()

    scattering_dict = {"scattering_type": args.scattering_type,
                       "scale_type": f"largest_scale_{args.largest_scale}",
                       "layers": [i+1 for i in range(args.num_layers)],
                       "moments" : [i+1 for i in range(args.highest_moment)]}
    
    main(args,scattering_dict)

    #Example : python classify_scattering.py --dataset=traffic --largest_scale=4 --sub_dataset=PEMS04 --scattering_type=blis --task_type=DAY


