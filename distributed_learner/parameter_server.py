class ParameterServer:
    def __init__(bind_port, param_dims, max_update_count):
        # Init server
        # Init params Nd-array
        # Init update rate : alpha
        pass

    def start_server():
        # bind port
        # start multithreaded server
        pass

    def stop_server():
        # return params
        # stop all active threads

    def set_update_rate(new_update_rate):
        # lock update rate variable
        # set update rate
        # unlock variable

    def _server_thread_function(thread_id):
        # lock update rate
        # lock param array
        # lock update_count
        # update param array
        # increment update counter, send update_count
        # send back new params if last updated sometime back
