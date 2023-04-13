import collections
import copy
import functools
import itertools
import logging
import operator
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Iterator, NamedTuple, Optional, OrderedDict, Set, Union

import torch._guards

import torch._logging

import torch.nn
from torch import fx
from torch._guards import (
    Checkpointable,
    Guard,
    GuardsCheckpointState,
    Source,
    TracingContext,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from . import config, logging as torchdynamo_logging, variables
from .backends.registry import CompiledFn, CompilerFn
from .bytecode_transformation import (
    create_call_function,
    create_instruction,
    Instruction,
    unique_id,
)
from .codegen import PyCodegen
from .exc import BackendCompilerFailed, unimplemented
from .guards import GuardBuilder
from .mutation_guard import is_dynamic_nn_module
from .side_effects import SideEffects
from .source import (
    ConstantSource,
    DeterministicAlgorithmsSource,
    is_constant_source,
    LocalSource,
    ParamBufferSource,
    ShapeEnvSource,
)
from .utils import (
    assert_no_fake_params_or_buffers,
    checkpoint_params,
    CleanupHook,
    clone_inputs,
    count_calls,
    counters,
    dynamo_timed,
    lazy_format_graph_code,
    lazy_format_graph_tabular,
    nnmodule_doc_url_msg,
    nnmodule_has_hooks,
    same,
)
from .variables.base import VariableTracker
from .variables.builder import GraphArg, TrackedFake, VariableBuilder, wrap_fx_proxy
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)

log = logging.getLogger(__name__)
graph_tabular_log = torch._logging.getArtifactLogger(__name__, "graph")
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code")


class OutputGraphState(NamedTuple):
    input_source_to_var: Dict[Source, VariableTracker]
    tracked_fakes: List[TrackedFake]
    guard_state: GuardsCheckpointState
    nn_modules: Optional[Dict[str, torch.nn.Module]]
    param_name_to_source: Optional[Dict[str, Source]]
    side_effects: SideEffects
    timestamp: int

    def diff(self, other: "OutputGraphState", *, prefix: str = "") -> Optional[str]:
        for k in self._fields:
            if k == "guard_state":
                r = self.guard_state.diff(other.guard_state)
                if r is not None:
                    return r
                continue
            elif k == "side_effects":
                r = self.side_effects.diff(other.side_effects)
                if r is not None:
                    return r
                continue

            sv = getattr(self, k)
            ov = getattr(other, k)
            if sv != ov:
                return f"{prefix}{k} mismatch: {sv} != {ov}"
        return None

    # Back compat .guards api
    @property
    def guards(self):
        return self.guard_state.dynamo_guards


@functools.lru_cache(None)
def _step_logger():
    return torchdynamo_logging.get_step_logger(log)


@dataclass
class GraphCompileReason:
    """Stores why a given output graph was compiled; i.e. what caused the graph break."""

    reason: str
    user_stack: List[traceback.FrameSummary]

    # Indicates if this was a graph compile reason due to graph break.
    graph_break: bool = True


def _get_gen_rand_values_fn(random_calls):
    def _gen_rand_values():
        return [fn(*args, **kwargs) for fn, args, kwargs in random_calls]

    return _gen_rand_values


class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""

    def __init__(self, nn_modules: Dict[str, torch.nn.Module]):
        super().__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)

    def __repr__(self):
        return "FakeRootModule(...)"


class WrapperBackend:
    def __init__(self, backend: CompilerFn, original_example_inputs):
        self.backend: CompilerFn = backend
        self.original_example_inputs = original_example_inputs

    @property
    def example_inputs(self):
        return clone_inputs(self.original_example_inputs)

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.restore = checkpoint_params(gm)
        self.gm = gm
        copy_gm = copy.deepcopy(self.gm)
        self.candidate = self.backend(copy_gm, self.original_example_inputs)

        if self.candidate is None or self.candidate is self.gm.forward:
            return self.gm.forward

        if not config.verify_correctness:
            return self.candidate

        # if verify_correctness=True
        try:
            correct = self.gm.forward(*self.example_inputs)
            result = self.candidate(*self.example_inputs)

            # TODO: replace `same` function with the one in testing
            if same(correct, result):
                return self.candidate

            raise RuntimeError(f"incorrect results of backend {self}")
            return self.gm.forward

        except Exception:
            log.exception("error in verify_correctness")
            raise
        finally:
            self.restore()


class OutputGraph(fx.Tracer, Checkpointable[OutputGraphState]):
    """
    Wrapper class to hold outputs of InstructionTranslator.  Mainly the
    generated fx.Graph.
    """

    def __init__(
        self,
        f_globals: Dict[str, Any],
        code_options: Dict[str, Any],
        compiler_fn: CompilerFn,
        root_tx,
        export: bool,
        export_constraints,
        frame_state,
    ):
        super().__init__()
        self.graph = torch.fx.Graph()
        # Map from graph input's `Source` to its `VariableTracker` to
        # de-duplicate graph inputs by source and reuse the tracker
        self.input_source_to_var: Dict[Source, VariableTracker] = {}
        self.export = export
        self.export_constraints = export_constraints
        self.frame_state = frame_state
        # In export mode, we force the shape_env to strictly disallow any constraining
        # of the user marked dynamic dims
        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(
                allow_scalar_outputs=config.capture_scalar_outputs,
                allow_dynamic_output_shape_ops=config.capture_dynamic_output_shape_ops,
            )
            if config.dynamic_shapes
            else None,
            # TODO (tmanlaibaatar) Remove this once we always lift params and buffers
            allow_non_fake_inputs=True if self.export else False,
        )
        self.tracing_context: TracingContext = TracingContext(fake_mode)
        if config.dynamic_shapes:
            # Register a SHAPE_ENV guard to make sure we setup shape guards
            # that show up in ShapeEnv
            self.guards.add(ShapeEnvSource().make_guard(GuardBuilder.SHAPE_ENV))

        self.guards.add(
            DeterministicAlgorithmsSource().make_guard(
                GuardBuilder.DETERMINISTIC_ALGORITHMS
            )
        )

        # tracked_fakes says where any tensor that was wrapped to fake came
        # from.  It is similar to GraphArg, in that all GraphArgs will get
        # will get added to TrackedFakes, but TrackedFakes also contains
        # GraphArgs that got pruned, and things like Tensor attributes which
        # aren't explicit graph inputs.  Used by shape guard
        self.tracked_fakes: List[TrackedFake] = []
        self.nn_modules: Optional[Dict[str, torch.nn.Module]] = dict()
        # Stores the full fqn of a param or buffer to the relevant source.
        self.param_name_to_source: Optional[Dict[str, Source]] = dict()
        self.side_effects = SideEffects()
        self.code_options = dict(code_options)
        self.output_instructions: List[Instruction] = []
        # used to track nodes that are added between calls of copy_graphstate
        # and restore_graphstate
        self.timestamp = 0
        # Node => computed real value (see utils.get_real_value)
        self.real_value_cache: Dict[fx.Node, torch.Tensor] = {}

        # Not checkpointed
        self.compiler_fn: CompilerFn = compiler_fn
        self.root_globals = f_globals
        self.root_tx = root_tx
        from torch._dynamo.symbolic_convert import InstructionTranslatorBase

        self._current_tx: List[InstructionTranslatorBase] = []
        self.cleanups: List[CleanupHook] = []
        self.should_exit = False
        self.random_values_var = None
        self.unspec_variable_map: Dict[str, UnspecializedPythonVariable] = {}

        # Map from graph input name to its placeholder proxy object, where the
        # map's keys give all current placeholder node names and can be used to
        # create unique node names
        self.input_name_to_proxy: OrderedDict[str, fx.Proxy] = collections.OrderedDict()

    @property
    def output(self):
        return self

    @property
    def fake_mode(self):
        return self.root_tx.fake_mode

    @property
    def shape_env(self):
        return self.tracing_context.fake_mode.shape_env

    @property
    def guards(self) -> Set[Guard]:
        return self.tracing_context.guards_context.dynamo_guards

    def push_tx(self, tx):
        self._current_tx.append(tx)

    def pop_tx(self):
        return self._current_tx.pop()

    @property
    def current_tx(self):
        return self.root_tx if not self._current_tx else self._current_tx[-1]

    def copy_graphstate(self) -> OutputGraphState:
        """Create a checkpoint of the current state by copying everything"""
        assert self.nn_modules is not None
        assert self.param_name_to_source is not None
        guards_graph_state = self.tracing_context.guards_context.copy_graphstate()
        state = OutputGraphState(
            dict(self.input_source_to_var),
            list(self.tracked_fakes),
            guards_graph_state,
            dict(self.nn_modules),
            dict(self.param_name_to_source),
            self.side_effects.clone(),
            self.timestamp,
        )
        self.timestamp += 1
        return state

    def restore_graphstate(self, state: OutputGraphState):
        """Restore a checkpoint created by self.copy_graphstate()"""
        (
            self.input_source_to_var,
            self.tracked_fakes,
            guards_state,
            self.nn_modules,
            self.param_name_to_source,
            self.side_effects,
            self.timestamp,
        ) = state
        self.tracing_context.guards_context.restore_graphstate(guards_state)
        # FX deepcopy doesn't work for a partially created graph, so just remove new nodes
        removed_nodes = 0
        for node in reversed(list(self.graph.nodes)):
            if node.meta["creation_timestamp"] > self.timestamp:
                # Erasing node alone does not remove the meta information
                # So, remove the help tensor explicitly
                if "example_value" in node.meta:
                    del node.meta["example_value"]
                self.remove_node(node)
                self.real_value_cache.pop(node, None)
                removed_nodes += 1
        log.debug("restore_graphstate: removed %s nodes", removed_nodes)

    def count_calls(self):
        return count_calls(self.graph)

    def get_submodule(self, keys):
        assert keys
        obj = self.nn_modules
        for k in keys.split("."):
            if isinstance(obj, dict):
                obj = obj[k]
            else:
                obj = getattr(obj, k)
        return obj

    def create_graph_input(self, name, type_expr=None):
        # unique
        if name in self.input_name_to_proxy:
            for i in itertools.count():
                candidate_name = f"{name}_{i}"
                if candidate_name not in self.input_name_to_proxy:
                    name = candidate_name
                    break

        if self.input_name_to_proxy:
            prev_name = next(reversed(self.input_name_to_proxy))
            ctx = self.graph.inserting_after(self.input_name_to_proxy[prev_name].node)
        else:
            ctx = self.graph.inserting_before(None)
        with ctx:
            proxy = self.create_proxy("placeholder", name, (), {}, type_expr=type_expr)
            self.input_name_to_proxy[name] = proxy
            return proxy

    def new_var(self, name="tmp"):
        existing = set(self.code_options["co_varnames"])
        for i in itertools.count():
            var = f"___{name}_{i}"
            if var not in existing:
                self.code_options["co_varnames"] += (var,)
                return var

    def update_co_names(self, name):
        """Ensure self.code_options.co_names contains name"""
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)

    def register_attr_or_module(
        self,
        target: Union[torch.nn.Module, torch.Tensor, Any],
        *names,
        **options,
    ):
        if is_dynamic_nn_module(target):
            return variables.UnspecializedNNModuleVariable(target, **options)

        options = dict(options)
        options["guards"] = set(options.get("guards", []))
        assert "source" in options
        source = options["source"]
        assert not isinstance(source, ParamBufferSource)

        if isinstance(target, torch.Tensor):
            if not is_constant_source(source):
                options["guards"].add(source.make_guard(GuardBuilder.TENSOR_MATCH))

            def wrap_name(module_key):
                assert self.param_name_to_source is not None
                self.param_name_to_source[module_key] = source
                return wrap_fx_proxy(
                    self.root_tx,
                    self.create_proxy("get_attr", module_key, tuple(), {}),
                    example_value=target,
                    **options,
                )

        elif isinstance(target, torch.nn.Module):
            assert isinstance(target, torch.nn.Module)
            if nnmodule_has_hooks(target, check_forward_hooks=True):
                torch._logging.warning_once(
                    log,
                    "nn.Module forward/_pre hooks are only partially supported, and were detected in your model. "
                    "In particular, if you do not change/remove hooks after calling .compile(), you can disregard this "
                    "warning, and otherwise you may need to set torch._dynamo.config.skip_nnmodule_hook_guards=False "
                    "to ensure recompiling after changing hooks."
                    f"{nnmodule_doc_url_msg} ",
                )
            if nnmodule_has_hooks(
                target, check_backward_hooks=True, check_state_dict_hooks=True
            ):
                torch._logging.warning_once(
                    log,
                    "nn.Module state_dict and backward hooks are not yet supported by torch.compile, "
                    f"but were detected in your model and will be silently ignored. {nnmodule_doc_url_msg}",
                )

            options["guards"].add(source.make_guard(GuardBuilder.NN_MODULE))

            def wrap_name(module_key):
                return NNModuleVariable(type(target), module_key, **options)

        elif isinstance(target, (torch.SymInt, torch.SymFloat)):
            # HACKY CODE REGION BEGIN
            # WE ARE PIGGYBACKING ON EXISTING INFRA TO REGISTER ATTRS
            # This ultimately gets written to self.nn_modules, which is unfortunate
            # Attrs that are tenors and symints and such need to be migrated to have their
            # own storage
            # alas, this is like this for now

            def wrap_name(module_key):
                return SymNodeVariable.create(
                    self,
                    self.create_proxy("get_attr", module_key, tuple(), {}),
                    sym_num=target,
                    **options,
                )

            # HACKY CODE REGION END
        else:

            def wrap_name(module_key):
                self.output.update_co_names(module_key)
                self.root_globals[module_key] = target
                return VariableBuilder(self, ConstantSource(source_name=module_key))(
                    target
                )

        assert self.nn_modules is not None
        for k, v in self.nn_modules.items():
            if v is target:
                # it already exists
                return wrap_name(k)
        # create a new unique name
        name = "_".join(map(str, names))
        # Strip the guard lookup L/G access
        name = re.sub(r"^[GL]\['?(.*?)'?\]$", r"\1", name)
        # e.g. replace abc.xyz[123].qkv with abc.xyz_123.qkv
        name = re.sub(r"\[(\d+)\]", r"_\g<1>", name)
        # e.g. replace abc.xyz_123.qkv with abc_xyz_123_qkv
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)

        if not name or not name[0].isalpha():
            name = "sub" + name
        base = name
        for i in itertools.count():
            if name not in self.nn_modules:
                self.nn_modules[name] = target
                if isinstance(target, torch.nn.Module):

                    def register_leaf_name(leaf_name):
                        assert self.param_name_to_source is not None
                        new_source = ParamBufferSource(source, leaf_name)
                        new_name = f"{name}.{leaf_name}"
                        self.param_name_to_source[new_name] = new_source

                    # annoying, but there are cases when we do not have parameters
                    # see test_nn_moduledict_contains
                    if hasattr(target, "_parameters"):
                        for leaf_name, _ in target.named_parameters(
                            remove_duplicate=False
                        ):
                            register_leaf_name(leaf_name)
                    if hasattr(target, "_buffers"):
                        for leaf_name, _ in target.named_buffers(
                            remove_duplicate=False
                        ):
                            register_leaf_name(leaf_name)

                return wrap_name(name)
            name = f"{base}_{i}"

        raise AssertionError("unreachable")

    def compile_subgraph(
        self, tx, partial_convert=False, reason: GraphCompileReason = None
    ):
        """
        Generate a subgraph to continue execution on user code.
        Automatically restore live variables.
        """
        assert reason is not None

        from .eval_frame import disable

        self.partial_convert = partial_convert
        self.compile_subgraph_reason = reason

        log.debug("COMPILING GRAPH due to %s", reason)

        if not all(block.can_restore() for block in tx.block_stack):
            unimplemented("compile_subgraph with block_depth != 0")

        prefix_insts: List[Instruction] = []
        if sys.version_info >= (3, 11):
            # prefix instructions (Python 3.11+)
            for inst in tx.prefix_insts:
                if inst.opname == "MAKE_CELL":
                    prefix_insts.append(
                        create_instruction("MAKE_CELL", argval=inst.argval)
                    )
                elif inst.opname == "COPY_FREE_VARS":
                    prefix_insts.append(
                        create_instruction(
                            "COPY_FREE_VARS", arg=len(tx.code_options["co_freevars"])
                        )
                    )
                else:
                    prefix_insts.append(inst)

        def append_prefix_insts():
            self.add_output_instructions(prefix_insts)
            prefix_insts.clear()

        for block in reversed(tx.block_stack):
            block.exit(tx)

        tx.prune_dead_locals()
        stack_values = list(tx.stack)
        assert self.nn_modules is not None
        root = FakeRootModule(self.nn_modules)

        # Add all the local vars to the "stack" so restore at the end
        restore_vars = []
        val_to_names: OrderedDict[
            VariableTracker, List[str]
        ] = collections.OrderedDict()
        if stack_values:
            val_to_names[stack_values[-1]] = list()
        for k, v in tx.symbolic_locals.items():
            # Note! this explicitly uses .local_name for matching
            # Failure to do so will cause spurious registrations in val_to_names.
            # This will in turn result in spurious variables showing up in the graph.
            # This was very tricky to debug. For an example, dump the graph at call_user_compiler
            # while running test_subgraphs.py
            if isinstance(v.source, LocalSource) and v.source.local_name == k:
                continue  # no need to restore initial state
            if v not in val_to_names:
                val_to_names[v] = list()
            val_to_names[v].append(k)
        for v in val_to_names.keys():
            restore_vars.extend(val_to_names[v])
            stack_values.extend([v] * len(val_to_names[v]))

        # to handle random calls
        if len(tx.random_calls) > 0:
            append_prefix_insts()
            random_calls_instructions = []
            self.random_values_var = self.new_var("random_values")
            rand_fn_name = unique_id("__gen_rand_values")
            rand_fn = disable(_get_gen_rand_values_fn(tx.random_calls))
            self.install_global(rand_fn_name, rand_fn)
            codegen = PyCodegen(tx, root)
            random_calls_instructions.extend(
                codegen.load_function_name(rand_fn_name, True)
            )
            random_calls_instructions.extend(create_call_function(0, False))
            random_calls_instructions.append(
                codegen.create_store(tx.output.random_values_var),
            )
            self.add_output_instructions(random_calls_instructions)

        if (
            stack_values
            and all(
                not isinstance(v, UnspecializedPythonVariable) for v in stack_values
            )
            and all(isinstance(x, TensorVariable) for x in stack_values)
            and len(set(stack_values)) == len(stack_values)
            and self.side_effects.is_empty()
        ):
            append_prefix_insts()
            # optimization to generate better code in a common case
            self.add_output_instructions(
                self.compile_and_call_fx_graph(tx, list(reversed(stack_values)), root)
                + [create_instruction("UNPACK_SEQUENCE", arg=len(stack_values))]
            )
        else:
            graph_output_var = self.new_var("graph_out")
            pass1 = PyCodegen(tx, root, graph_output_var)
            self.side_effects.codegen_save_tempvars(pass1)
            pass1.foreach(stack_values)
            self.side_effects.codegen_update_mutated(pass1)

            # one more time now that we have established tempvars
            pass2 = PyCodegen(
                tx,
                root,
                graph_output_var,
                tempvars={val: None for val, count in pass1.uses.items() if count > 1},
            )
            self.side_effects.codegen_save_tempvars(pass2)
            pass2.foreach(stack_values)
            self.side_effects.codegen_update_mutated(pass2)

            output = []
            if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
                output.extend(
                    self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
                )

                if len(pass2.graph_outputs) != 0:
                    output.append(pass2.create_store(graph_output_var))
                else:
                    output.append(create_instruction("POP_TOP"))
            append_prefix_insts()
            self.add_output_instructions(output + pass2.get_instructions())

        # restore all the live local vars
        self.add_output_instructions(
            [PyCodegen(tx).create_store(var) for var in reversed(restore_vars)]
        )

    @torch._guards.TracingContext.clear_frame()
    def compile_and_call_fx_graph(self, tx, rv, root):
        """
        Generate code from self.graph and return the Instruction()s to
        call that generated code.
        """
        from .eval_frame import disable

        assert isinstance(rv, list)
        assert isinstance(root, FakeRootModule)
        for output in rv:
            self.guards.update(output.guards)

        self.create_node(
            "output", "output", (self.create_arg(tuple(x.as_proxy() for x in rv)),), {}
        )
        self.remove_unused_graphargs()
        ncalls = count_calls(self.graph)
        counters["stats"]["calls_captured"] += ncalls

        # free a bit of memory
        for node in self.graph.nodes:
            if "example_value" in node.meta:
                del node.meta["example_value"]
        self.real_value_cache.clear()

        gm = fx.GraphModule(root, self.graph)
        gm.recompile()
        gm.compile_subgraph_reason = self.compile_subgraph_reason
        name = unique_id("__compiled_fn")

        assert_no_fake_params_or_buffers(gm)
        compiled_fn = self.call_user_compiler(gm)
        compiled_fn = disable(compiled_fn)

        counters["stats"]["unique_graphs"] += 1
        self.install_global(name, compiled_fn)

        graph_code_log.debug("%s", lazy_format_graph_code(name, gm))
        graph_tabular_log.debug("%s", lazy_format_graph_tabular(name, gm))

        cg = PyCodegen(tx)
        cg.make_call_generated_code(name)
        return cg.get_instructions()

    @property
    def placeholders(self) -> Iterator[fx.Node]:
        r = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                r.append(node)
                continue
            break
        return r

    @property
    def graphargs(self) -> List[GraphArg]:
        return [node.meta['grapharg'] for node in self.placeholders]

    @dynamo_timed(phase_name="backend_compile")
    def call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:
        tot = 0
        placeholders = []
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                tot += 1
            if node.op == "placeholder":
                placeholders.append(node)
        torch._dynamo.utils.increment_op_count(tot)
        for pl in placeholders:
            arg = pl.meta['grapharg']
            # TODO: Why isn't this stored in meta :think:
            pl._dynamo_source = arg.source

        gm._param_name_to_source = self.param_name_to_source

        try:
            name = (
                self.compiler_fn.__name__
                if hasattr(self.compiler_fn, "__name__")
                else ""
            )
            _step_logger()(logging.INFO, f"calling compiler function {name}")
            compiler_fn = self.compiler_fn
            # WrapperBackend needs real inputs, for now, to verify correctness
            if config.verify_correctness:
                compiler_fn = WrapperBackend(compiler_fn, self.example_inputs())

            # NOTE: [Real Tensors in Accuracy Evaluation]
            #
            # Today, tensors are passed to backends as fake at compile time. See the .fake_example_inputs()
            # call to compiler_fn below. At runtime, backends use real tensors.
            #
            # This should be a strong invariant we hold across all backends,
            # and generally, it is. However, for accuracy evaluation, we need real tensors at compile time,
            # for now, due to the unfortunate setup described below.
            #
            # Due to the nature of how we invoke comparison as a backend in two different ways:
            #
            # (1) Less bad, but still worth rewriting, WrapperBackend above, which takes
            # real inputs for its ctor. see the config.verify_correctnes above.
            #
            # (2) More bad, and very worth rewriting, the minifier installs accuracy comparison as
            # a true backend, and therefore needs to be compiled with real inputs. This is made trickier
            # by the fact that the minifier will spawn new processes during minification. As such, we have
            # created a global flag, MINIFIER_SPAWNED, that should be set IF AND ONLY IF this run was spawned
            # as part of accuracy minification. This flag is not a contract, and ideally will not be here long.
            #
            # The longer term PoR is to:
            # (A) Rewrite the minifier accuracy evaluation and verify_correctness code to share the same
            # correctness and accuracy logic, so as not to have two different ways of doing the same thing.
            #
            # (B) Refactor minifier accuracy backend to do its comparison fully at runtime, so as not to need to
            # pass real tensors to it at compile time.
            is_top_level_minifying = (
                config.repro_after is not None and config.repro_level == 4
            )
            if torch._dynamo.debug_utils.MINIFIER_SPAWNED or is_top_level_minifying:
                # Disable the tracing context so we don't pick up the ambient
                # fake tensor mode
                with torch._guards.tracing(None):
                    compiled_fn = compiler_fn(gm, self.example_inputs())
            elif config.DO_NOT_USE_legacy_non_fake_example_inputs:
                compiled_fn = compiler_fn(gm, self.example_inputs())
            else:
                compiled_fn = compiler_fn(gm, self.fake_example_inputs())
            _step_logger()(logging.INFO, f"done compiler function {name}")
            assert callable(compiled_fn), "compiler_fn did not return callable"
        except Exception as e:
            raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
                e.__traceback__
            ) from None
        return compiled_fn

    def fake_example_inputs(self) -> List[torch.Tensor]:
        result = []
        for arg in self.graphargs:
            if arg.fake_tensor is not None:
                result.append(arg.fake_tensor)
            else:
                # Fallback, in case fake_tensor was not set
                # Particularly for graph args that are not tensors
                result.append(arg.example)
        return result

    def example_inputs(self) -> List[torch.Tensor]:
        result = []
        for arg in self.graphargs:
            result.append(arg.example)
        return result

    def remove_unused_graphargs(self) -> None:
        # Miniature DCE pass, but only for obviously trivial operations
        for node in reversed(list(self.graph.nodes)):
            if len(list(node.users)) == 0:
                if node.op == "get_attr":
                    self.remove_node(node)
                elif node.op == "call_function" and node.target is operator.getitem:
                    self.remove_node(node)

        for i, node in enumerate(self.placeholders):
            if not node.users:
                log.debug("REMOVE UNUSED GRAPHARG %s", node.meta["grapharg"].source.name())
                # I'm not really sure why you need to delete these from the
                # node since the node is going to get removed
                if "example_value" in node.meta:
                    del node.meta["example_value"]
                del node.meta["grapharg"]
                self.remove_node(node)
                self.real_value_cache.pop(node, None)

    def add_output_instructions(self, prefix: List[Instruction]) -> None:
        """
        We call this on the creation of a new compiled subgraph that is inserted
        before user code.
        """
        self.output_instructions.extend(prefix)
        self.should_exit = True

    def install_global(self, name, value) -> None:
        self.cleanups.append(CleanupHook.create(self.root_globals, name, value))

    def cleanup(self) -> None:
        # There is a reference cycle between tracer and OutputGraph, causing
        # some of the tensor objects to be held alive for longer than necessary.

        self.root_tx = None

        # Note: generated fx graph will hold a reference to the nn_module,
        # So depending on the backend they may not be released
        self.nn_modules = None
        self.param_name_to_source = None

        for node in self.graph.nodes:
            if "example_value" in node.meta:
                del node.meta["example_value"]
            if "grapharg" in node.meta:
                del node.meta["grapharg"]
        self.real_value_cache.clear()
        self.input_name_to_proxy.clear()
        self.side_effects.keepalive = []

    def create_proxy(
        self,
        kind,
        target,
        args,
        kwargs,
        name=None,
        type_expr=None,
        proxy_factory_fn=None,
    ):
        rv = super().create_proxy(
            kind, target, args, kwargs, name, type_expr, proxy_factory_fn
        )

        # append stack trace to fx node
        tx = self.current_tx

        nn_module_stack = tx.nn_module_stack
        if nn_module_stack:
            rv.node.meta["nn_module_stack"] = nn_module_stack.copy()

        if kind in {"call_function", "call_method"}:
            rv.node.meta["source_fn"] = target
        elif kind == "call_module":
            # For modules we store the class
            rv.node.meta["source_fn"] = rv.node.meta["nn_module_stack"][target][1]

        frame_summaries: List[traceback.FrameSummary] = []
        while tx:
            frame_summaries.append(tx.frame_summary())
            tx = getattr(tx, "parent", None)
        # Reverse the frame_summaries, such that the innermost frame is at the last
        frame_summaries.reverse()

        # official from_list stub doesn't have new-style type
        msgs = traceback.StackSummary.from_list(frame_summaries).format()  # type: ignore[arg-type]
        rv.node.stack_trace = "".join(msgs)

        return rv

    def create_node(self, *args, **kwargs):
        node = super().create_node(*args, **kwargs)
        node.meta["creation_timestamp"] = self.timestamp
        return node

    # Note: we did not override erase_node since
    # we call self.graph.erase_node elsewhere
    def remove_node(self, node):
        self.graph.erase_node(node)
        self.input_name_to_proxy.pop(node.name, None)
