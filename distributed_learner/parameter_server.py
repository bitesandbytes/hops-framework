import socket
import threading
import sys
import numpy.random.normal as rng

class ParameterServer:
    def __init__(self, update_port, update_rate=0.01, sync_port, num_params):
        # Init TCP server
        self.upd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sync_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.update_port = update_port
        self.sync_port = sync_port

        self.update_rate = update_rate

        self.params = rng(size=num_params)*1e-2

    def start_servers(self):
        # bind port
        self.upd_server_addr = upd_sock.bind('localhost', update_port)
        print >> sys.stderr, 'Starting update server on %s, port %s.', upd_server_addr
        self.sync_server_addr = sync_sock.bind('localhost', sync_port)
        print >> sys.stderr, 'Starting sync server on %s, port %s.', sync_server_addr

        # start multithreaded server
        self.upd_thread = threading.Thread(target=self._update_server_thread, args=None)
        self.sync_thread = threading.Thread(target=self._sync_server_thread, args=None)

    def _update_server_thread(self):
        # listen for incoming connections
        self.upd_sock.listen(1)

        while True:
            print >> sys.stderr, 'Update Server listening'
            conn, cli_addr = self.upd_sock.accept()
            print >> sys.stderr, 'Update Server accepted connection from ', cli_addr
            # Receive params
            # lock self.params
            # update self.params
            # unlock self.params
            conn.close()

        pass

    def _sync_server_thread(self):
        # listen for incoming connections
        self.sync_sock.listen(1)

        while True:
            print >> sys.stderr, 'Sync Server listening'
            conn, cli_addr = self.sync_sock.accept()
            print >> sys.stderr, 'Sync Server accepted connection from ', cli_addr
            # lock self.params
            # send self.params
            # unlock self.params
            conn.close()

        pass
