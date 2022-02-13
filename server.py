import configparser
import numpy as np

from numpy.lib.npyio import load
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.models.ncf.dataset import Dataset as NCFDataset



def ml_fedavg(client_wts):
    pass


def distribute_client_data(data, items, users, n_clients, loader, num_neg):
    sample_frac = 0.3
    client_data = []
    for c in range(n_clients):
        print("Sampling data for client " + str(c))
        df = data.sample(frac=sample_frac, random_state=seed)
        c_uids, c_iids = loader.get_samples(df, items)
        user_input, item_input, labels = loader.get_train_instances(c_uids, c_iids, num_neg, len(items))
        c_data = {
            'df': df,
            'items': items,
            'users': users,
            'uids': c_uids,
            'iids': c_iids,
            'user_input': user_input,
            'item_input': item_input,
            'labels': labels
        }
        client_data.append(c_data)
    return client_data

def initialize_clients(client_data, weights, epochs, batch_size, seed):
    from client import Client
    clients = []
    for i in range(len(client_data)):
        c_d = client_data[i]
        c = Client(c_d, epochs, batch_size, seed)
        c.set_weights(weights)
        clients.append(c)
    return clients

def train_server(seed, epochs, batch_size, rounds, n_clients):

    from dataset import Loader
    from NCF import NeuMF

    num_neg=4
    loader = Loader()
    df, items, users = loader.load_dataset()

    server_df = df.sample(frac=0.2, random_state=seed)
    s_uids, s_iids = loader.get_samples(server_df, items)
    s_user_input, s_item_input, s_labels = loader.get_train_instances(s_uids, s_iids, num_neg, len(items))
    s_user_input = np.array(s_user_input).reshape(-1,1)
    s_item_input = np.array(s_item_input).reshape(-1,1)
    s_labels = np.array(s_labels).reshape(-1,1)

    client_data = distribute_client_data(df, items, users, n_clients, loader, num_neg)

    
    server_model = NeuMF(len(users), len(items))
    server_wt = server_model.get_weights()
    server_model.set_weights(server_wt)
    # server_model.fit(s_user_input, s_item_input, s_labels, epochs, batch_size)
    
    clients = initialize_clients(client_data, server_wt, epochs, batch_size, seed)
    

    # for r in range(rounds):
    #     client_wts = []
    #     client_item_profiles = []

    #     for client in clients:
    #         client.set_weights(server_wt)
    #         client.fit()
    #         client_wts.append(client.get_weights())

    #     aggregate_wt = ml_fedavg(client_wts)
        


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    ml_datasize = config['DEFAULT']['MOVIELENS_DATA_SIZE']
    epochs = int(config['DEFAULT']['EPOCHS'])
    batch_size = int(config['DEFAULT']['BATCH_SIZE'])
    seed = int(config['DEFAULT']['SEED'])
    n_clients = int(config['DEFAULT']['N_CLIENTS'])
    rounds = int(config['DEFAULT']['COMMUNICATION_ROUNDS'])

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    df = movielens.load_pandas_df(
        size=ml_datasize,
        header=["userID", "itemID", "rating", "timestamp"]
    )

    train_server(seed, epochs, batch_size, rounds, n_clients)
    