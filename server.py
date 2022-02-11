import configparser
from recommenders.datasets import movielens

def distribute_client_data(n_clients, ml_datasize):
    sample_frac = 1 / n_clients
    df = movielens.load_pandas_df(
        size=ml_datasize,
        header=["userID", "itemID", "rating", "timestamp"]
    )
    client_datas = []
    for c in range(n_clients):
        client_datas.append(df.sample(frac=sample_frac))
    return client_datas

def train_server():
    pass


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    ml_datasize = config['DEFAULT']['MOVIELENS_DATA_SIZE']
    epochs = int(config['DEFAULT']['EPOCHS'])
    batch_size = int(config['DEFAULT']['BATCH_SIZE'])
    seed = int(config['DEFAULT']['SEED'])
    n_clients = int(config['DEFAULT']['N_CLIENTS'])
    client_datas = distribute_client_data(n_clients)
    