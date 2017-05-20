class ParameterClient:
    def __init__(self, server_ip, update_port, sync_port, update_rate):
        self.server_ip = server_ip
        self.update_port = update_port
        self.sync_port = sync_port
        self.sync = update_rate
        self.update_rate = update_rate
        # Init client sockets
        # Init client Nd-arrays

    def send_updates(self, update_deltas):
        # send update deltas

        # If not sync'ed in self.sync_in cycles, sync.
        self.sync_in -= 1

        if self.sync_in == 0:
            pass # ask for params
