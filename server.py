import configparser
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.models.ncf.dataset import Dataset as NCFDataset

from NCF import NCF


def distribute_client_data(data, n_clients, ml_datasize):
    sample_frac = 1 / n_clients
    client_datas = []
    for c in range(n_clients):
        client_datas.append(data.sample(frac=sample_frac))
    return client_datas

def train_server(client_datas, server_d, seed, epochs, batch_size):
    server_data = NCFDataset(train=server_d, seed=seed)
    server_model = NCF(
        n_users=server_data.n_users,
        n_items=server_data.n_items,
        n_factors=4,
        layer_sizes=[16, 8, 4],
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        verbose=10,
        seed=seed
    )
    server_wt = server_model.get_weights()
    # print(server_wt)
    server_model.set_weights(server_wt)

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    ml_datasize = config['DEFAULT']['MOVIELENS_DATA_SIZE']
    epochs = int(config['DEFAULT']['EPOCHS'])
    batch_size = int(config['DEFAULT']['BATCH_SIZE'])
    seed = int(config['DEFAULT']['SEED'])
    n_clients = int(config['DEFAULT']['N_CLIENTS'])

    df = movielens.load_pandas_df(
        size=ml_datasize,
        header=["userID", "itemID", "rating", "timestamp"]
    )

    client, server_data = python_chrono_split(df, 0.75)
    client_datas = distribute_client_data(client, n_clients, ml_datasize)
    
    train_server(client_datas, server_data, seed, epochs, batch_size)
    