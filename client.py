class Client:
    def __init__(
        self,
        data,
        epochs,
        batch_size,
        seed,
    ):
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._set_model()

    def _set_model(self):
        from NCF import NeuMF

        self.model = NeuMF(len(self.data['users']), len(self.data['items']))

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def fit(self, epochs, batch_size):
        self.model.fit(self.data['user_input'], self.data['item_input'], self['labels'], epochs, batch_size)
    