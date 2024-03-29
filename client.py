class Client:
    def __init__(
        self,
        data,
        epochs,
        batch_size,
        seed,
        id
    ):
        self.data = data
        self.epochs = epochs 
        self.batch_size = batch_size
        self.seed = seed
        self.id = id
        self._set_model()

    def _set_model(self):
        from NCF import NeuMF
        from metrics import Metric

        self.model = NeuMF(len(self.data['users']), len(self.data['items']), self.id)
        self.metric = Metric()


    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def get_data_size(self):
        return self.data['length']

    def fit(self, epochs, batch_size):
        
        self.model.fit(self.data['user_input'], self.data['item_input'], self.data['labels'], epochs, batch_size)
    
    def validate(self):
        import numpy as np
        hit_lst = self.metric.evaluate_top_k(self.data['df_neg'], self.data['df_test'], self.model.model, K=10)
        hit_rate = np.mean(hit_lst)
        print("Client hit rate: ", hit_rate)

    def print_user_embeddings(self):
        print("User Embedding GMF: \n")
        print(self.model.user_embedding_gmf.get_weights()[0][0])
        print("\n")
        # print("User Embedding MLP: \n")
        # print(self.model.user_embedding_mlp.get_weights())