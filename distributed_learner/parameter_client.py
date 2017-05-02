class ParameterClient:
    def __init__(self, server_ip, server_port, param_dims):
        self.server_ip = server_ip
        self.server_port = server_port
        # Init client sockets
        # Init client Nd-arrays

    def send_updates(self, update_deltas):
        # send update deltas
        # if send_count == K, receive new params
