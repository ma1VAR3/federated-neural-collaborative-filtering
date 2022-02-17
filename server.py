import configparser
import numpy as np
import pandas as pd

from recommenders.datasets import movielens



def ml_fedavg(client_wts, client_data_sizes):
    
    average_wts = {}
    fracs = []
    n = 0
    for size in client_data_sizes:
        n += size
    for size in client_data_sizes:
        fracs.append(size/n)

    #Averaging MLP weights
    average_weight_mlp=[]
    for index1 in range(len(client_wts[0]['mlp'])):
        layer_weights=[]
        for index2 in range(len(client_wts)):
            weights=client_wts[index2]['mlp'][index1]
            layer_weights.append(weights)
        average_weight=np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_mlp.append(average_weight)

    #Averaging output layer weights
    average_weight_output = []
    for index1 in range(len(client_wts[0]['output'])):
        outputs = []
        for index2 in range(len(client_wts)):
            weights=client_wts[index2]['output'][index1]
            outputs.append(weights)
        average_output=np.mean(np.array([x for x in outputs]), axis=0)
        average_weight_output.append(average_output)

    #Averaging item embeddings for GMF
    average_item_profile_gmf = []
    for index1 in range(len(client_wts[0]['item_embedding_gmf'])):
        item_gmf = []
        for index2 in range(len(client_wts)):
            weights=client_wts[index2]['item_embedding_gmf'][index1]
            item_gmf.append(weights)
        average_item_gmf=np.mean(np.array([x for x in item_gmf]), axis=0)
        average_item_profile_gmf.append(average_item_gmf)

    #Averaging item embeddings for MLP
    average_item_profile_mlp = []
    for index1 in range(len(client_wts[0]['item_embedding_mlp'])):
        item_mlp = []
        for index2 in range(len(client_wts)):
            weights=client_wts[index2]['item_embedding_mlp'][index1]
            item_mlp.append(weights)
        average_item_mlp=np.mean(np.array([x for x in item_mlp]), axis=0)
        average_item_profile_mlp.append(average_item_mlp)


    average_wts['mlp'] = average_weight_mlp
    average_wts['output'] = average_weight_output
    average_wts['item_embedding_gmf'] = average_item_profile_gmf
    average_wts['item_embedding_mlp'] = average_item_profile_mlp
    
    return average_wts


def distribute_client_data(data, items, users, n_clients, loader, num_neg):
    sample_frac = 0.10
    test_frac = 0.04
    client_data = []
    for c in range(n_clients):
        print("Sampling data for client " + str(c))
        r = np.random.random_sample()
        while r > 0.2:
            r = np.random.random_sample()
        sample = sample_frac + r
        df = data.sample(frac=sample, random_state=seed)
        length = df.shape[0]
        print("Length ", length)
        c_uids, c_iids = loader.get_samples(df, items)
        user_input, item_input, labels = loader.get_train_instances(c_uids, c_iids, num_neg, len(items))
        user_input = np.array(user_input).reshape(-1,1)
        item_input = np.array(item_input).reshape(-1,1)
        labels = np.array(labels).reshape(-1,1)

        df_test = data.sample(frac=test_frac, random_state=seed)
        test_uids, test_iids = loader.get_samples(df_test, items)
        df_test = df_test.groupby(['user_id']).first()
        df_test['user_id'] = df_test.index
        df_test = df_test[['user_id', 'item_id', 'rating']]
        df_test = df_test.reset_index(drop=True)
        df_neg = get_negatives(test_uids, test_iids, items, df_test)

        c_data = {
            'df': df,
            'items': items,
            'users': users,
            'uids': c_uids,
            'iids': c_iids,
            'user_input': user_input,
            'item_input': item_input,
            'labels': labels,
            'df_test': df_test,
            'df_neg': df_neg,
            'length': length
        }
        client_data.append(c_data)
    return client_data

def initialize_clients(client_data, weights, epochs, batch_size, seed):
    from client import Client
    clients = []
    for i in range(len(client_data)):
        c_d = client_data[i]
        c = Client(c_d, epochs, batch_size, seed, i)
        c.set_weights(weights)
        clients.append(c)
    return clients

def get_negatives(uids, iids, items, df_test):

        negativeList = []
        test_u = df_test['user_id'].values.tolist()
        test_i = df_test['item_id'].values.tolist()

        test_ratings = list(zip(test_u, test_i))  # test (user, item)
        zipped = set(zip(uids, iids))             # train (user, item)

        for (u, i) in test_ratings:

            negatives = []
            negatives.append((u, i))
            for t in range(100):
                j = np.random.randint(len(items))     # neg_item j 
                while (u, j) in zipped:              
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
    
    server_model = NeuMF(len(users), len(items), "server")
    server_wt = server_model.get_weights()
    # server_model.set_weights(server_wt)
    # server_model.fit(s_user_input, s_item_input, s_labels, epochs, batch_size)
    
    clients = initialize_clients(client_data, server_wt, epochs, batch_size, seed)
    
    metric = Metric()

    hits = []
    for r in range(rounds):
        print("\n\n"+"="*30+" Starting round " + str(r+1)+" "+"="*30+"\n")
        client_wts = []
        client_data_sizes = []
        cid = 0
        for client in clients:
            print("="*15+" Training client " + str(cid)+" "+"="*15)
            client.set_weights(server_wt)
            client.fit(epochs, batch_size)
            client_wts.append(client.get_weights())
            client_data_sizes.append(client.get_data_size())
            client.validate()
            cid += 1
        
        server_wt = ml_fedavg(client_wts, client_data_sizes)
        server_model.set_weights(server_wt)
        
        hit_lst = metric.evaluate_top_k(df_neg, df_test, server_model.model, K=10)
        hit = np.mean(hit_lst)

        print("\n "+"*"*10+" Server side hit rate: ", hit, " "+"*"*10)

        hits.append(hit)
    
    return hits

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

    hits = train_server(seed, epochs, batch_size, rounds, n_clients)
    
    with open('hit_rate.npy', 'wb') as f:
        np.save(f, hits)
    f.close()