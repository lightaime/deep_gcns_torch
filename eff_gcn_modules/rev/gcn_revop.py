"""This module is implemented by Guohao Li based on MemCNN @ Copyright (c) 2018 Sil C. van de Leemput under MIT license."""

import numpy as np
import torch
import torch.nn as nn

use_context_mans = True

try:
    pytorch_version_one_and_above = int(torch.__version__[0]) > 0
except TypeError:
    pytorch_version_one_and_above = True


class InvertibleCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, fn_inverse, keep_input, num_bwd_passes, preserve_rng_state, num_inputs, *inputs_and_weights):
        # store in context
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.keep_input = keep_input
        ctx.weights = inputs_and_weights[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.preserve_rng_state = preserve_rng_state
        ctx.num_inputs = num_inputs
        inputs = inputs_and_weights[:num_inputs]

        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*inputs)

        ctx.input_requires_grad = [element.requires_grad for element in inputs]

        with torch.no_grad():
            # Makes a detached copy which shares the storage
            x = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    x.append(element.detach())
                else:
                    x.append(element)
            outputs = ctx.fn(*x)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Detaches y in-place (inbetween computations can now be discarded)
        detached_outputs = tuple([element.detach_() for element in outputs])

        # clear memory from inputs
        # only clear memory of node features
        if not ctx.keep_input:
            if not pytorch_version_one_and_above:
                # PyTorch 0.4 way to clear storage for node features
                inputs[0].data.set_()
            else:
                # PyTorch 1.0+ way to clear storage for node features
                inputs[0].storage().resize_(0)

        # store these tensor nodes for backward pass
        ctx.inputs = [inputs] * num_bwd_passes
        ctx.outputs = [detached_outputs] * num_bwd_passes

        return detached_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # pragma: no cover
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("InvertibleCheckpointFunction is not compatible with .grad(), please use .backward() if possible")
        # retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError("Trying to perform backward on the InvertibleCheckpointFunction for more than "
                               "{} times! Try raising `num_bwd_passes` by one.".format(ctx.num_bwd_passes))
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # recompute input if necessary
        if not ctx.keep_input:
            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  Restore the surrounding state
            # when we're done.
            rng_devices = []
            if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
                rng_devices = ctx.fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
                # recompute input
                with torch.no_grad():
                    # edge_index and edge_emb
                    inputs_inverted = ctx.fn_inverse(*(outputs+inputs[1:]))
                    # clear memory from outputs
                    if not pytorch_version_one_and_above:
                        # PyTorch 0.4 way to clear storage
                        for element in outputs:
                            element.data.set_()
                    else:
                        # PyTorch 1.0+ way to clear storage
                        for element in outputs:
                            element.storage().resize_(0)

                    if not isinstance(inputs_inverted, tuple):
                        inputs_inverted = (inputs_inverted,)
                    if pytorch_version_one_and_above:
                        for element_original, element_inverted in zip(inputs, inputs_inverted):
                            element_original.storage().resize_(int(np.prod(element_original.size())))
                            element_original.set_(element_inverted)
                    else:
                        for element_original, element_inverted in zip(inputs, inputs_inverted):
                            element_original.set_(element_inverted)

        # compute gradients
        with torch.set_grad_enabled(True):
            detached_inputs = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    detached_inputs.append(element.detach())
                else:
                    detached_inputs.append(element)
            detached_inputs = tuple(detached_inputs)
            for det_input, requires_grad in zip(detached_inputs, ctx.input_requires_grad):
                det_input.requires_grad = requires_grad
            temp_output = ctx.fn(*detached_inputs)
        if not isinstance(temp_output, tuple):
            temp_output = (temp_output,)

        filtered_detached_inputs = tuple(filter(lambda x: x.requires_grad,
                                               detached_inputs))
        gradients = torch.autograd.grad(outputs=temp_output,
                                        inputs=filtered_detached_inputs + ctx.weights,
                                        grad_outputs=grad_outputs)

        # Setting the gradients manually on the inputs and outputs (mimic backwards)
        filtered_inputs = list(filter(lambda x: x.requires_grad,
                                      inputs))

        input_gradients = []
        i = 0
        for rg in ctx.input_requires_grad:
            if rg:
                input_gradients.append(gradients[i])
                i += 1
            else:
                input_gradients.append(None)

        gradients = tuple(input_gradients) + gradients[-len(ctx.weights):]

        return (None, None, None, None, None, None) + gradients


class InvertibleModuleWrapper(nn.Module):
    def __init__(self, fn, keep_input=False, keep_input_inverse=False, num_bwd_passes=1,
                 disable=False, preserve_rng_state=False):
        """
        The InvertibleModuleWrapper which enables memory savings during training by exploiting
        the invertible properties of the wrapped module.

        Parameters
        ----------
            fn : :obj:`torch.nn.Module`
                A torch.nn.Module which has a forward and an inverse function implemented with
                :math:`x == m.inverse(m.forward(x))`

            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            num_bwd_passes :obj:`int`, optional
                Number of backward passes to retain a link with the output. After the last backward pass the output
                is discarded and memory is freed.
                Warning: if this value is raised higher than the number of required passes memory will not be freed
                correctly anymore and the training process can quickly run out of memory.
                Hence, The typical use case is to keep this at 1, until it raises an error for raising this value.

            disable : :obj:`bool`, optional
                This will disable using the InvertibleCheckpointFunction altogether.
                Essentially this renders the function as `y = fn(x)` without any of the memory savings.
                Setting this to true will also ignore the keep_input and keep_input_inverse properties.

            preserve_rng_state : :obj:`bool`, optional
                Setting this will ensure that the same RNG state is used during reconstruction of the inputs.
                I.e. if keep_input = False on forward or keep_input_inverse = False on inverse. By default
                this is False since most invertible modules should have a valid inverse and hence are
                deterministic.

        Attributes
        ----------
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

        """
        super(InvertibleModuleWrapper, self).__init__()
        self.disable = disable
        self.keep_input = keep_input
        self.keep_input_inverse = keep_input_inverse
        self.num_bwd_passes = num_bwd_passes
        self.preserve_rng_state = preserve_rng_state
        self._fn = fn

    def forward(self, *xin):
        """Forward operation :math:`R(x) = y`

        Parameters
        ----------
            *xin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *y.

        """
        if not self.disable:
            y = InvertibleCheckpointFunction.apply(
                self._fn.forward,
                self._fn.inverse,
                self.keep_input,
                self.num_bwd_passes,
                self.preserve_rng_state,
                len(xin),
                *(xin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            y = self._fn(*xin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(y, tuple) and len(y) == 1:
            return y[0]
        return y

    def inverse(self, *yin):
        """Inverse operation :math:`R^{-1}(y) = x`

        Parameters
        ----------
            *yin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *x.

        """
        if not self.disable:
            x = InvertibleCheckpointFunction.apply(
                self._fn.inverse,
                self._fn.forward,
                self.keep_input_inverse,
                self.num_bwd_passes,
                self.preserve_rng_state,
                len(yin),
                *(yin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            x = self._fn.inverse(*yin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
        return x

# To consider:  maybe get_device_states and set_device_states should reside in
# torch/random.py?
#
# get_device_states and set_device_states cannot be imported from
# torch.utils.checkpoint, since it was not
# present in older versions, so we include a copy here.
def get_device_states(*args):
      # This will not error out if "arg" is a CPU tensor or a non-tensor type
      # because
      # the conditionals short-circuit.
      fwd_gpu_devices = list(set(arg.get_device() for arg in args
                            if isinstance(arg, torch.Tensor) and arg.is_cuda))

      fwd_gpu_states = []
      for device in fwd_gpu_devices:
          with torch.cuda.device(device):
              fwd_gpu_states.append(torch.cuda.get_rng_state())

      return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states):
      for device, state in zip(devices, states):
          with torch.cuda.device(device):
              torch.cuda.set_rng_state(state)
