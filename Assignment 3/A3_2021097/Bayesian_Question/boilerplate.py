#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    train_df = pd.read_csv("train_data.csv")
    val_df = pd.read_csv("validation_data.csv")
    return train_df, val_df
    pass

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    # dfhot, dfnum = bn.df2onehot(df)
    features = df.columns
    edges=[]
    for i in range(len(features)):
        for j in range(i+1,len(features)):
            edges.append((features[i],features[j]))
        
    DAG = bn.make_DAG(edges)
    model = bn.parameter_learning.fit(DAG, df)
    # print("model done")
    # Plot
    G = bn.plot(DAG)
    bn.plot_graphviz(DAG)
    return model
    pass

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model
    with open('base_model.pkl', 'rb') as file:
        model_A = pickle.load(file)
    
    model_sl = bn.independence_test(model_A, df,prune=True)
    edges_new = model_sl['model_edges']
    DAG = bn.make_DAG(edges_new)
    prune_model = bn.parameter_learning.fit(DAG, df)
    bn.plot(prune_model)
    return prune_model
    pass

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    with open('base_model.pkl', 'rb') as file:
        model_A = pickle.load(file)
    # model_new = bn.parameter_learning.fit(model_A, df, methodtype = 'hc')
    param = bn.structure_learning.fit(df,methodtype = 'hc')
    opt_model = bn.parameter_learning.fit(param, df)
    bn.plot(param)

    return opt_model
    pass

def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname, 'wb') as file:
        pickle.dump(model, file)
    pass

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

# import time

# def func_timer(start_time, end_time,func):
#     print(f"Time taken by {func} is {end_time-start_time} seconds")


def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    # start_time = time.time()
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)
    # end_time = time.time()
    # func_timer(start_time,end_time,"make_network")

    # Create and save pruned model
    # start_time = time.time()
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)
    # end_time = time.time()
    # func_timer(start_time,end_time,"make_pruned_network")

    # Create and save optimized model
    
    # start_time = time.time()
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)
    # end_time = time.time()
    # func_timer(start_time,end_time,"make_optimized_network")

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()

