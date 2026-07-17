from typing import List, Optional, Sequence, TypeVar

import ir.AST as T

TNode = TypeVar("TNode")


def seq_from_list(items: Sequence[TNode]) -> Optional[TNode]:
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    node: TNode = items[-1]
    for item in reversed(items[:-1]):
        node = T.Seq(item, node)  # type: ignore[assignment]
    return node


def sequentialize_ast(node: T.ASTNode) -> T.ASTNode:
    if isinstance(node, T.Block):
        seq = seq_from_list([sequentialize_ast(stmt) for stmt in node.stmts])
        return seq if seq is not None else T.Block([])
    if isinstance(node, T.Loop):
        return T.Loop(node.start, node.end, node.tile_name, node.loop_var, sequentialize_ast(node.body))
    if isinstance(node, T.If):
        then_branch = sequentialize_ast(node.then_branch)
        else_branch = sequentialize_ast(node.else_branch) if node.else_branch else None
        return T.If(node.cond, then_branch, else_branch)
    if isinstance(node, T.Let):
        return T.Let(node.tensor, node.value, sequentialize_ast(node.body))
    return node


def sequentialize_primfunc(primfunc: T.PrimFunc) -> T.PrimFunc:
    return T.PrimFunc(
        name=primfunc.name,
        input_tensors=primfunc.input_tensors,
        output_tensor=primfunc.output_tensor,
        spatial_axes=primfunc.spatial_axes,
        root_node=sequentialize_ast(primfunc.root_node),
        allocated_tensors=primfunc.allocated_tensors,
    )


def sequentialize_calls(calls: List[T.PrimFuncCall]) -> Optional[object]:
    return seq_from_list(calls)


def sequentialize_main_calls(calls: List[T.PrimFuncCall]) -> List[T.PrimFuncCall]:
    updated: List[T.PrimFuncCall] = []
    for call in calls:
        updated.append(
            T.PrimFuncCall(
                primfunc=sequentialize_primfunc(call.primfunc),
                out_var_tensor=call.out_var_tensor,
                input_tensors=call.input_tensors,
                call_index=call.call_index,
            )
        )
    return updated


def sequentialize_main_func(main_func: T.MainFunc) -> T.MainFunc:
    return T.MainFunc(
        calls=sequentialize_main_calls(main_func.calls),
        input_tensors=main_func.input_tensors,
        output_tensors=main_func.output_tensors,
        intermediate_tensors=main_func.intermediate_tensors,
    )
