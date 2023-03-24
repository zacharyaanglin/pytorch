from functools import partial

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional, _wrap_all_tensors_to_functional, functionalize
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
    unwrap_proxy,
)
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)
from torch.utils._pytree import tree_flatten
from ._cond import _has_potential_branch_input_alias, _has_potential_branch_input_mutation, UnsupportedAliasMutationException


map = HigherOrderOperator("map")

def map_without_ad(f, xs, *args):
    _ = torch._C._AutoDispatchBelowAutograd()
    return map(f, xs, *args)

class MapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fw_graph, bw_graph, args_spec, *flat_args):
        ctx.save_for_backward(*flat_args)
        ctx.bw_graph = bw_graph
        ctx._args_spec = args_spec
        xs, args = pytree.tree_unflatten(flat_args, args_spec)
        out = map_without_ad(fw_graph, xs, *args)
        flat_out, out_spec =  pytree.tree_flatten(out)
        return out_spec, *flat_out
    
    @staticmethod
    def backward(ctx, _, *flat_grads):
        xs, args = pytree.tree_unflatten(ctx.saved_tensors, ctx._args_spec)
        grads = map_without_ad(ctx.bw_graph, (xs, flat_grads), *args)
        return None, None, None, *pytree.tree_flatten(grads)[0]

def trace_map(proxy_mode, func_overload, f, xs, *args):
    if not all(isinstance(o, torch.Tensor) for o in args):
        raise ValueError("map() positional args must be a list of tensors")
    
    def check_tensor(arg):
        if not isinstance(arg, torch.Tensor):
            raise ValueError("map() operands must be a list of tensors")
        if len(arg.shape) == 0 or arg.shape[0] == 0:
            raise ValueError("map() cannot be traced with scalar tensors or zero dimension tensors")
        
    pytree.tree_map(check_tensor, xs)
    flat_xs, _ = pytree.tree_flatten(xs)
    leading_dim_size = flat_xs[0].shape[0]

    xs_pytrees = _unstack_pytree(xs)
    with disable_proxy_modes_tracing():
        body_graph = make_fx(f)(xs_pytrees[0], *args)

    next_name = None
    i = 0
    while not next_name:
        candidate = f"body_graph_{i}"
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate

    proxy_mode.tracer.root.register_module(next_name, body_graph)
    node_args = (body_graph, xs, *args)
    proxy_args = pytree.tree_map(partial(unwrap_proxy, proxy_mode), node_args)
    out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {},
                                               name="map")
    example_outs = body_graph(xs_pytrees[0], *args)
    expanded_outs = pytree.tree_map(lambda t: t.expand(leading_dim_size, *t.shape), example_outs)
    # # Implementation notes: we need to use new_empty() + copy_() here instead of stack() directly
    # # because stack([...]) takes a fixed size list which will specialize dynamic shape here.
    # # Meanwhile we want to preserve the looped over dimension as symbolic shape, such that:
    # # ys: Tensor[s0, ...] = map(xs: Tensor[s0, ...], *args)
    # out = outs[0].new_empty([xs.shape[0], *outs[0].shape])
    # out.copy_(torch.stack(outs))
    return track_tensor_tree(expanded_outs, out_proxy, constant=None, tracer=proxy_mode.tracer)

def _unstack_pytree(xs):
    flat_xs, inspec = pytree.tree_flatten(xs)
    assert all([isinstance(xs, torch.Tensor) for xs in flat_xs]), f"Leaves of xs must be Tensor {flat_xs}"
    assert all([xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs]), f"Leaves of xs must have same leading dimension size {flat_xs}"
    a = list(zip(*flat_xs))
    pytrees = []
    for tuple in a:
        pytrees.append(pytree.tree_unflatten(tuple, inspec))
    return pytrees

def _stack_pytree(pytrees):
    flat_out = []
    out_spec = None
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        flat_out.append(flat_pt)
    b = list(zip(*flat_out))
    stacked_out = []
    for leaves in b:
        if all([leave is not None for leave in leaves]):
            stacked_out.append(torch.stack(leaves))
        else:
            stacked_out.append(None)
    return pytree.tree_unflatten(stacked_out, out_spec)

@map.py_impl(DispatchKey.CUDA)
@map.py_impl(DispatchKey.CPU)
def map_impl(f, xs, *args):
    mode = _get_current_dispatch_mode()
    assert (mode is None), "Mode should never be enabled for CPU/CUDA keyOne of the differentiated Tensors"
    pytrees = []
    for inp in _unstack_pytree(xs):
        pytrees.append(f(inp, * args))
    return _stack_pytree(pytrees)


@map.py_impl(DispatchKey.Autograd)
def map_autograd(f, xs, *args):
    def fw_bw_joint(xs_and_grad, *args):
        example_xs, example_grad = xs_and_grad
        fw_ret = f(example_xs, *args)
        flat_ret, _ = pytree.tree_flatten(fw_ret)
        flat_xs, _ = pytree.tree_flatten(example_xs)
        # To simplify the handling of input and output tensors in torch.autograd.grad function, 
        # we 1. create dummy placeholders for inputs and setting allow_unused=True.
        # the gradient of these dummy tensors will be None.
        flat_inputs = (input if input.requires_grad else torch.empty(1, requires_grad=True) for input in (*flat_xs, *args) )
        return torch.autograd.grad(flat_ret, flat_inputs, example_grad, allow_unused=True)

    example_xs = _unstack_pytree(xs)[0]
    example_flat_out, _ = pytree.tree_flatten(f(example_xs, *args))
    example_grad = [torch.ones_like(out) for out in example_flat_out]

    fw_graph = make_fx(f, tracing_mode="fake")(example_xs, *args)
    # exmaple_xs and example_grad are both mapped over the leading dimension, we group them
    # together so as to follow map operator's usage pattern
    joint_graph = make_fx(fw_bw_joint, tracing_mode="fake")((example_xs, example_grad), *args)

    print("fw_graph:")
    fw_graph.print_readable()
    print("joint_graph:")
    joint_graph.print_readable()

    print(f"fw_ret:{fw_graph(example_xs, *args)}")
    print(f"bw_ret:{joint_graph((example_xs, example_grad), *args)}")

    # Autograd.Function only handles flattend inputs and produces flattend outputs.
    flat_args, args_spec = pytree.tree_flatten((xs, args))
    out_spec, *flat_outs= MapAutogradOp.apply(fw_graph, joint_graph, args_spec, *flat_args)
    return pytree.tree_unflatten(flat_outs, out_spec)


@map.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(f, xs, *args):
    print("map forward proxy torch dispatch")
    mode = _get_current_dispatch_mode()
    assert (mode is not None), "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        res = trace_map(mode, map, f, xs, *args)
    return res


@map.py_impl(FakeTensorMode)
def map_fake_tensor_mode(f, xs, *args):
    print("map_fake_tensor")
    outs = [f(x, *args) for x in xs]
    return outs[0].new_empty([xs.shape[0], *outs[0].shape])

@map.py_impl(torch._C._functorch.TransformType.Functionalize)
def map_functionalize(interpreter, f, xs, *args):
    print("map_functionalize")
    """
    Functionalization implementation for torch.map. Currently:
      1. We don't allow any input mutation inside the map function
      2. Our check for above condition is not exhaustive
    """
    reapply_views = interpreter.functionalize_add_back_views()
    mode = 'mutations_and_views' if reapply_views else 'mutations'
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    unwrapped_args = _unwrap_all_tensors_from_functional(args, reapply_views=reapply_views)

    functional_map_fn = functionalize(f, remove=mode)

    with interpreter.lower():
        inputs = (unwrapped_xs,) + unwrapped_args
        if _has_potential_branch_input_mutation(functional_map_fn, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is mutating the input!"
            )

        if _has_potential_branch_input_alias(functional_map_fn, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is aliasing the input!"
            )

        map_return = map(functional_map_fn, unwrapped_xs, *unwrapped_args)
        return _wrap_all_tensors_to_functional(map_return, level=interpreter.level())

# TODO(voz) Make this automatic for keys, this is very ugly atm
map.fallthrough(DispatchKey.PythonDispatcher)
map.fallthrough(DispatchKey.PythonTLSSnapshot)
map.fallthrough(DispatchKey.ADInplaceOrView)
map.fallthrough(DispatchKey.BackendSelect)
map.fallthrough(DispatchKey.AutocastCPU)