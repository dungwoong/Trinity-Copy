from dataclasses import dataclass
from typing import List, Union

__all__ = [
    "TensorInfo",
    "PrimFunc",
    "ASTNode",
    "Tensor",
    "VarRef",
    "Tile",
    "TileOffset",
    "ConstTile",
    "FullTile",
    "Elem",
    "Index",
    "Load",
    "Store",
    "Const",
    "Loop",
    "Seq",
    "Block",
    "If",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Max",
    "Min",
    "Exp",
    "Sqr",
    "Sqrt",
    "Sigmoid",
    "Matmul",
    "Take",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "Concat",
    "Broadcast",
    "Permute3",
    "Squeeze",
    "Unsqueeze",
    "GenericBinary",
    "GenericCall",
    "Arange",
    "Cast",
    "PrimFuncCall",
    "MainFunc",
]

@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    dtype: str
    
@dataclass
class PrimFunc:
    name: str
    input_tensors: List[TensorInfo]
    output_tensor: TensorInfo
    spatial_axes: List[str]
    root_node: "ASTNode"
    allocated_tensors: List[TensorInfo]

# Base Type
@dataclass
class ASTNode:
    pass

# --- Data Access & Structure ---
@dataclass
class Tensor(ASTNode):
    name: str

@dataclass
class VarRef(ASTNode):
    name: str  # scalar loop var reference

@dataclass
class Tile(ASTNode):
    name: str  # loop var name or tile name

@dataclass
class TileOffset(ASTNode):
    name: str
    offset: int

@dataclass
class ConstTile(ASTNode):
    start_index: int
    interval: int

@dataclass
class FullTile(ASTNode):
    pass

@dataclass
class Elem(ASTNode):
    name: str

@dataclass
class Index(ASTNode):
    indices: List[ASTNode]  # Box<[Id]>

@dataclass
class Load(ASTNode):
    tensor: Tensor
    index: Index

@dataclass
class Store(ASTNode):
    tensor: Tensor
    value: ASTNode
    index: Index

@dataclass
class Const(ASTNode):
    value: Union[int, float, str]

# --- Control Flow ---
@dataclass
class Loop(ASTNode):
    start: ASTNode
    end: ASTNode
    tile_name: str
    loop_var: str
    body: ASTNode  # or List[ASTNode] if implicit block

@dataclass
class Seq(ASTNode):
    left: object
    right: object

@dataclass
class Block(ASTNode):
    # User definition didn't strictly have a 'Block', but we need it 
    # to hold multiple statements inside a loop.
    stmts: List[ASTNode]

@dataclass
class If(ASTNode):
    cond: ASTNode
    then_branch: ASTNode
    else_branch: ASTNode = None

@dataclass
class Let(ASTNode):
    tensor: ASTNode
    value: ASTNode      
    body: ASTNode

# --- Arithmetic (Scalar) ---
@dataclass
class Add(ASTNode):
    left: ASTNode
    right: ASTNode

@dataclass
class Sub(ASTNode):
    left: ASTNode
    right: ASTNode

@dataclass
class Mul(ASTNode): 
    left: ASTNode
    right: ASTNode

@dataclass
class Div(ASTNode):
    left: ASTNode
    right: ASTNode

@dataclass
class Max(ASTNode):
    left: ASTNode
    right: ASTNode

@dataclass
class Min(ASTNode):
    left: ASTNode
    right: ASTNode

@dataclass
class Exp(ASTNode):
    val: ASTNode

@dataclass
class Sqr(ASTNode):
    val: ASTNode

@dataclass
class Sqrt(ASTNode):
    val: ASTNode

@dataclass
class Sigmoid(ASTNode):
    val: ASTNode

# --- Tensor Operations ---
@dataclass
class Matmul(ASTNode): # Represents "@" (Tensor Matmul)
    left: ASTNode
    right: ASTNode

@dataclass
class Take(ASTNode):  # Represents gather/take with 1D indices
    data: ASTNode
    indices: ASTNode
    axis: int
    index: ASTNode

@dataclass
class ReduceSum(ASTNode):
    val: ASTNode
    axis: int

@dataclass
class ReduceMax(ASTNode):
    val: ASTNode
    axis: int

@dataclass
class ReduceMin(ASTNode):
    val: ASTNode
    axis: int

@dataclass
class Concat(ASTNode):
    a: ASTNode
    b: ASTNode
    axis: int

@dataclass
class Broadcast(ASTNode):  # Represents broadcast(a, axis)
    val: ASTNode
    axis: int

@dataclass
class Permute3(ASTNode):
    val: ASTNode
    d0: int
    d1: int
    d2: int

@dataclass
class Squeeze(ASTNode):
    val: ASTNode
    axis: int

@dataclass
class Unsqueeze(ASTNode):
    val: ASTNode
    axis: int

# --- Fallbacks for missing ops in user list ---
@dataclass
class GenericBinary(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode

@dataclass
class GenericCall(ASTNode):
    func_name: str
    args: List[ASTNode]

@dataclass
class Arange(ASTNode):
    axis: str

@dataclass
class Cast(ASTNode):
    dtype: str
    val: ASTNode

@dataclass
class PrimFuncCall:
    primfunc: PrimFunc
    out_var_tensor: TensorInfo
    input_tensors: List[TensorInfo]
    call_index: int

@dataclass
class MainFunc:
    calls: List[PrimFuncCall]
    input_tensors: List[TensorInfo]
    output_tensors: List[TensorInfo]
    intermediate_tensors: List[TensorInfo]
    
