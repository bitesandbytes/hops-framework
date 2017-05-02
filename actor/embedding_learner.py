import tensorflow as tf
import tflearn

# Learns an embedding function f:SxG -> Z
# |S|<=20, |G|<= 20
# Try 10-dim Z-space for embedding
class EmbeddingLearner(object):
    # inp_dim = (S+G)
    # emd_dim = Z
    def __init__(self, inp_dim, emb_dim, learning_rate):
        self.input_dim = inp_dim
        self.embedding_dims = emb_dim

        # init TF network
        # config = [inp_dim, inp_dim+emb_dim, emb_dim, inp_dim+emb_dim, inp_dim]
        # Build encoder
        encoder = tflearn.input_data(shape=[None, inp_dimm])
        encoder = tflearn.fully_connected(encoder, (inp_dim+emb_dim))
        encoder = tflearn.fully_connected(encoder, emb_dim)
        decoder = tflearn.fully_connected(encoder, inp_dim+emb_dim)
        decoder = tflearn.fully_connected(decoder, inp_dim)

        # Set objective function and optimizer
        net = tflearn.regression(decoder, optimizer='adam', learning_rate=learning_rate,
                         loss='mean_square', metric=None)
        self.model = tflearn.DNN(net, tensorboard_verbose=0)

    # state_goal_matrix = Nx(S+G)
    def learn(state_goal_matrix, val_ratio=0.1, batch_size=256):
        # TODO(bitesandbytes) : split data into train:test using (1-val_ratio):val_ratio
        self.model.fit(state_goal_matrix, state_goal_matrix, n_epoch=10, validation_set=(testX, testX), run_id="auto_encoder", batch_size=batch_size)

        # Get only encoder in a new model
        # Re-use weights from self.model training
        self.encoder_model = tflearn.DNN(encoder, session=self.model.session)

    # state_goal_matrix = Nx(S+G)
    # returns embed_matrix = NxZ
    def embed(state_goal_matrix):
        return self.encoder_model.predict(state_goal_matrix)
