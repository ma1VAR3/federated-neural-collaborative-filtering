import configparser
import numpy as np
import pandas as pd

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
        user_input = np.array(user_input).reshape(-1,1)
        item_input = np.array(item_input).reshape(-1,1)
        labels = np.array(labels).reshape(-1,1)
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

def get_negatives(uids, iids, items, df_test):

        negativeList = []
        test_u = df_test['user_id'].values.tolist()
        test_i = df_test['item_id'].values.tolist()

        test_ratings = list(zip(test_u, test_i))  # test (user, item)세트
        zipped = set(zip(uids, iids))             # train (user, item)세트

        for (u, i) in test_ratings:

            negatives = []
            negatives.append((u, i))
            for t in range(100):
                j = np.random.randint(len(items))     # neg_item j 1개 샘플링
                while (u, j) in zipped:               # j가 train에 있으면 다시뽑고, 없으면 선택
                    j = np.random.randint(len(items))
                negatives.append(j)
            negativeList.append(negatives) # [(0,pos), neg, neg, ...]

        df_neg = pd.DataFrame(negativeList)

        return df_neg


def train_server(seed, epochs, batch_size, rounds, n_clients):

    from dataset import Loader
    from NCF import NeuMF
    from metrics import Metric

    num_neg=4
    loader = Loader()
    df, items, users = loader.load_dataset()

    server_df = df.sample(frac=0.2, random_state=seed)
    s_uids, s_iids = loader.get_samples(server_df, items)
    s_user_input, s_item_input, s_labels = loader.get_train_instances(s_uids, s_iids, num_neg, len(items))
    s_user_input = np.array(s_user_input).reshape(-1,1)
    s_item_input = np.array(s_item_input).reshape(-1,1)
    s_labels = np.array(s_labels).reshape(-1,1)

    df_test = server_df.copy(deep=True)
    df_test = df_test.groupby(['user_id']).first()
    df_test['user_id'] = df_test.index
    df_test = df_test[['user_id', 'item_id', 'rating']]
    df_test = df_test.reset_index(drop=True)

    df_neg = get_negatives(s_uids, s_iids, items, df_test)

    client_data = distribute_client_data(df, items, users, n_clients, loader, num_neg)
    
    server_model = NeuMF(len(users), len(items))
    server_wt = server_model.get_weights()
    # server_model.set_weights(server_wt)
    # server_model.fit(s_user_input, s_item_input, s_labels, epochs, batch_size)
    
    clients = initialize_clients(client_data, server_wt, epochs, batch_size, seed)
    
    metric = Metric()

    hits = []
    for r in range(rounds):

        client_wts = []
        client_item_profiles = []

        for client in clients:
            client.set_weights(server_wt)
            client.fit()
            client_wts.append(client.get_weights())

        # server_wt = ml_fedavg(client_wts)
        server_model.set_weights(server_wt)
        
        hit_lst = metric.evaluate_top_k(df_neg, df_test, server_model, K=10)
        hit = np.mean(hit_lst)
        hits.append(hit)

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
    