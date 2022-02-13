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

def initialize_clients(client_data, weights, epochs, batch_size, seed):
    from client import Client
    clients = []
    for i in range(len(client_data)):
        c_d = client_data[i]
        c = Client(c_d, epochs, batch_size, seed, "client"+str(i))
        # c.set_weights(weights)
        # clients.append(c)
    return clients


def ml_fedavg(client_wts):
    pass

def train_server(client_data, server_d, seed, epochs, batch_size, rounds):

    import tensorflow as tf
    import logging
    
    tf.compat.v1.disable_eager_execution()
    logger = logging.getLogger(__name__)
    tf.compat.v1.set_random_seed(seed)

    tf.compat.v1.reset_default_graph()

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    # set TF Session
    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    )
    # parameters initialization
    sess.run(tf.compat.v1.global_variables_initializer())

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
        seed=seed,
        prefix="server"
    )
    server_wt = server_model.get_weights()
    # print(server_wt)
    server_model.set_weights(server_wt)

    clients = initialize_clients(client_data, server_wt, epochs, batch_size, seed)
    

    for r in range(rounds):
        client_wts = []
        client_item_profiles = []

        for client in clients:
            client.set_weights(server_wt)
            client.fit()
            client_wts.append(client.get_weights())

        aggregate_wt = ml_fedavg(client_wts)
        


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    ml_datasize = config['DEFAULT']['MOVIELENS_DATA_SIZE']
    epochs = int(config['DEFAULT']['EPOCHS'])
    batch_size = int(config['DEFAULT']['BATCH_SIZE'])
    seed = int(config['DEFAULT']['SEED'])
    n_clients = int(config['DEFAULT']['N_CLIENTS'])
    rounds = int(config['DEFAULT']['COMMUNICATION_ROUNDS'])

    




    df = movielens.load_pandas_df(
        size=ml_datasize,
        header=["userID", "itemID", "rating", "timestamp"]
    )

    client, server_data = python_chrono_split(df, 0.75)
    client_data = distribute_client_data(client, n_clients, ml_datasize)
    train_server(client_data, server_data, seed, epochs, batch_size, rounds)
    