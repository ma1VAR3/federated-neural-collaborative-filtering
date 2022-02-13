class Client:
    def __init__(
        self,
        dataset,
        epochs,
        batch_size,
        seed,
        prefix
    ):
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.prefix = prefix
        self._set_model()

    def _set_model(self):
        from recommenders.models.ncf.dataset import Dataset as NCFDataset
        from NCF import NCF

        self.data = NCFDataset(train=self.dataset, seed=self.seed)
        self.model = NCF(
            n_users=self.data.n_users,
            n_items=self.data.n_items,
            n_factors=4,
            layer_sizes=[16, 8, 4],
            n_epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=1e-3,
            verbose=10,
            seed=self.seed
        )

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def fit(self):
        self.model.fit(self.data)
    