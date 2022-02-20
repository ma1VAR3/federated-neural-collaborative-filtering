import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Concatenate, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model

class NeuMF:

    def __init__(self, user_num, item_num, id):

        latent_features = 8
        self.id = id
        # Input
        user = Input(shape=(1,), dtype='int32')
        item = Input(shape=(1,), dtype='int32')

        # User embedding for GMF
        gmf_user_embedding = Embedding(user_num, latent_features, input_length=user.shape[1], name="gmf_user_embedding_client_"+str(self.id))
        gmf_user_embedding_inp = gmf_user_embedding(user)
        gmf_user_embedding_inp = Flatten()(gmf_user_embedding_inp)

        # Item embedding for GMF
        gmf_item_embedding = Embedding(item_num, latent_features, input_length=item.shape[1], name="gmf_item_embedding_client_"+str(self.id))
        gmf_item_embedding_inp = gmf_item_embedding(item)
        gmf_item_embedding_inp = Flatten()(gmf_item_embedding_inp)

        # User embedding for MLP
        mlp_user_embedding = Embedding(user_num, 32, input_length=user.shape[1], name="mlp_user_embedding_client_"+str(self.id))
        mlp_user_embedding_inp = mlp_user_embedding(user)
        mlp_user_embedding_inp = Flatten()(mlp_user_embedding_inp)

        # Item embedding for MLP
        mlp_item_embedding = Embedding(item_num, 32, input_length=item.shape[1], name="mlp_item_embedding_client_"+str(self.id))
        mlp_item_embedding_inp = mlp_item_embedding(item)
        mlp_item_embedding_inp = Flatten()(mlp_item_embedding_inp)

        self.user_embedding_mlp = mlp_user_embedding
        self.user_embedding_gmf = gmf_user_embedding
        self.item_embedding_mlp = mlp_item_embedding
        self.item_embeddin_gmf = gmf_item_embedding
        # GMF layers
        gmf_mul =  Multiply()([gmf_user_embedding_inp, gmf_item_embedding_inp])

        # MLP layers
        mlp_concat = Concatenate()([mlp_user_embedding_inp, mlp_item_embedding_inp])
        mlp_dropout = Dropout(0.2)(mlp_concat)

        # Layer1
        mlp_layer_1 = Dense(units=64, activation='relu', name='mlp_layer1')(mlp_dropout)  # (64,1)
        mlp_dropout1 = Dropout(rate=0.2, name='dropout1')(mlp_layer_1)                    # (64,1)
        mlp_batch_norm1 = BatchNormalization(name='batch_norm1')(mlp_dropout1)            # (64,1)

        # Layer2
        mlp_layer_2 = Dense(units=32, activation='relu', name='mlp_layer2')(mlp_batch_norm1)  # (32,1)
        mlp_dropout2 = Dropout(rate=0.2, name='dropout2')(mlp_layer_2)                        # (32,1)
        mlp_batch_norm2 = BatchNormalization(name='batch_norm2')(mlp_dropout2)                # (32,1)

        # Layer3
        mlp_layer_3 = Dense(units=16, activation='relu', name='mlp_layer3')(mlp_batch_norm2)  # (16,1)

        # Layer4
        mlp_layer_4 = Dense(units=8, activation='relu', name='mlp_layer4')(mlp_layer_3)       # (8,1)

        # merge GMF + MLP
        merged_vector = tf.keras.layers.concatenate([gmf_mul, mlp_layer_4])

        # Output layer
        output_layer = Dense(1, kernel_initializer='lecun_uniform', name='output_layer', activation='sigmoid') # 1,1 / h(8,1)
        output_layer_out = output_layer(merged_vector)
        output_dummy = Dense(1, kernel_initializer='lecun_uniform', name='output_layer', activation='sigmoid')(mlp_layer_4)

        self.output = output_layer

        # Model
        self.model = Model([user, item], output_layer_out)
        self.mlp = Model([user, item], output_dummy)
        self.model.compile(optimizer= 'adam', loss= 'binary_crossentropy')

    def get_model(self):
        model = self.model
        return model

    def get_weights(self):
        mlp_wt = self.mlp.get_weights()
        item_mlp = self.item_embedding_mlp.get_weights()
        item_gmf = self.item_embeddin_gmf.get_weights()
        output = self.output.get_weights()
        wts = {
            "item_embedding_mlp": item_mlp,
            "item_embedding_gmf": item_gmf,
            "mlp": mlp_wt,
            "output": output
        }
        return wts

    def set_weights(self, weights):
        self.mlp.set_weights(weights['mlp'])
        self.output.set_weights(weights['output'])
        self.item_embedding_mlp.set_weights(weights['item_embedding_mlp'])
        self.item_embeddin_gmf.set_weights(weights['item_embedding_gmf'])

    def fit(self, user_data, item_data, labels, epochs, batch_size):
        self.model.fit([user_data, item_data], labels, epochs=epochs, batch_size=batch_size)
