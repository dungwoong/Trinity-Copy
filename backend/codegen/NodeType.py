from enum import Enum

class NodeType(Enum):
    PLOOP = "ploop"  # Parallel loop
    SLOOP = "sloop"  # Sequential loop
    INPUT = "input"
    OUTPUT = "output"
    TILE = "tile"
    FULLTILE = "fulltile"
    DUMMY = "dummy"
    INDEX = "index"
    LOAD = "load"
    STORE = "store"
    SEQ = "seq"
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    LE = "<="
    MATMUL = "@"
    EXP = "exp"
    SQR = "sqr"
    SQRT = "sqrt"
    SIGMOID = "sigmoid"
    ERF = "erf"
    CAST = "cast"
    ABS = "abs"
    MAX = "max"
    MIN = "min"
    RMAX = "rmax"
    RMIN = "rmin"
    RSUM = "rsum"
    CONCAT = "concat"
    BCAST = "bcast"
    NUM = "num"
    VAR = "var"
    ELEM = "elem"  # Element-wise indexing
    PERMUTE3 = "permute3"  # 3D permutation
    TRANSPOSE = "transpose"
    SQUEEZE = "squeeze"  # Remove dimension
    UNSQUEEZE = "unsqueeze"  # Add dimension
    CONST_TILE = "const_tile"  # Constant range tile [start:start+size]
    TENSOR = "tensor"  # Intermediate tensor
