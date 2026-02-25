import numpy as np
import time
import torch

from .dynamic_batch import GeneratorArg, DynamicBatchGenerator


def torch_to_numpy(t):
    if t.dtype == torch.bfloat16:
        # return t.half().detach().cpu().numpy()
        return t.float().detach().cpu().numpy()
    return t.detach().cpu().numpy()


class SessionGenerator(object):
    def __init__(self, impl: DynamicBatchGenerator):
        self._impl = impl
        self._c_generator = impl._c_generator
        self._session_id = f"sess_{time.time():.3f}"
        self._chunk_pos = 0
        self._first_chunk = True

    def feed(
            self,
            input_ids,
            input_embeddings,
    ):
        # convert inputs
        assert input_ids is not None or input_embeddings is not None
        if input_embeddings is not None:
            if not isinstance(input_embeddings, np.ndarray):
                input_embeddings = torch_to_numpy(input_embeddings)
            assert input_embeddings.ndim == 2
            num_tokens = input_embeddings.shape[0]
        if input_ids is not None:
            num_tokens = len(input_ids)
        if input_ids is None:
            input_ids = [0] * input_embeddings.shape[0]

        arg = GeneratorArg(
            max_length=1,  # set max_length=1 to mimic encode only
            output_hidden_states=-1)
        arg.set_session_info(session_id=self._session_id,
                             session_continue=not self._first_chunk,
                             sess_chunk_pos=self._chunk_pos)
        self._chunk_pos += num_tokens
        self._first_chunk = False

        c_task = self._impl.to_c_task(input_ids, arg=arg)
        if input_embeddings is not None:
            c_task.set_input_embeddings(input_embeddings)

        req_out = self._impl.generate_c(c_task, arg)

        return req_out

    def close(self):
        self._c_generator.close_session(self._session_id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
