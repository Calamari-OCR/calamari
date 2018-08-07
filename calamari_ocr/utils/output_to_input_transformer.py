class OutputToInputTransformer:
    def __init__(self, data_processing, backend):
        self.data_processing = data_processing
        self.backend = backend

    def local_to_global(self, x, data_proc_params):
        x = self.backend.output_to_input_position(x)
        if self.data_processing:
            x = self.data_processing.local_to_global_pos(x, data_proc_params)
        return x