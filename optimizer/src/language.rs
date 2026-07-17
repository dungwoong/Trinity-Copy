use crate::shape::{Dimension, ShapeTracker, TensorShape};
use crate::utils::*;
use egg::*;
use std::cell::RefCell;
use std::collections::HashSet;

thread_local! {
    pub static SHAPE_TRACKER: RefCell<ShapeTracker> = RefCell::new(ShapeTracker::new());
}

define_language! {
    pub enum TileLang {

        "loop" = Loop([Id; 5]), // loop start end tile_n loop_var body
        "dloop" = DLoop([Id; 5]), // dloop start end tile_n loop_var body => loop that only fusion rule can be applied
        "tmp_loop" = TLoop([Id; 6]), // tmp_loop start end tile_n loop_var1 loop_var2 body => we need to rebuild this node during modify()
        "input" = Input(Id),    // name of tensor
        "output" = Output(Id), // name of final output tensor of the tensor program
        "tensor" = Tensor(Id), // Represent a name of intermediate tensor created by operation

        "tile" = Tile(Id), // tile n = [n:n+tile_n], tile_n is the tile size of loop that is related to n
        "const_tile" = ConstTile([Id; 2]), // (const_tile start_index interval) Represent a tile index of tensor. Start index and interval should be constant
        "fulltile" = FullTile,  // fulltile = [:]. Full tile of axis
        "elem" = Elem(Id), // elem n = [n//tile_n:n//tile_n+1], tile_n is the tile size of loop that is related to n
        "index" = Index(Box<[Id]>), // index (tile n) (tile m) ... = [m:m+tile_m, n:n+tile_n, ...]
        "load" = Load([Id; 2]), // (load A index) => A
        "store" = Store([Id; 3]), // store A val index ...
        "seq" = Seq([Id; 2]),   // seq body1 body2 ...

        "const" = Const(Id), // (const N) meaning that the N is a constant, not variable

        "+" = Add([Id; 2]), // a + b
        "-" = Sub([Id; 2]), // a - b
        "*" = Mul([Id; 2]), // a * b (elementwise)
        "/" = Div([Id; 2]), // a / b
        "<=" = Le([Id; 2]), // a <= b
        "max" = Max([Id; 2]), // max(a, b)
        "min" = Min([Id; 2]), // min(a, b)
        "exp" = Exp(Id), // exp(a)
        "@" = Matmul([Id; 2]), // a @ b (matrix multiplication)
        "rsum" = ReduceSum([Id; 2]), // reduce_sum(a, axis)
        "rmin" = ReduceMin([Id; 2]), // reduce_min(a, axis)
        "rmax" = ReduceMax([Id; 2]), // reduce_max(a, axis)
        "sqr" = Sqr(Id), // square(a)
        "sqrt" = Sqrt(Id), // sqrt(a)
        "sigmoid" = Sigmoid(Id), // sigmoid(a)
        "erf" = Erf(Id), // erf(a)
        "abs" = Abs(Id), // abs(a)
        "cast" = Cast([Id; 2]), // cast(dtype, a)

        "concat" = Concat([Id; 3]), // concat(a, b, axis)
        "bcast" = Broadcast([Id; 2]), // broadcast(a, axis)

        "transpose" = Transpose(Id),
        "permute3" = Permute3([Id; 4]), // permute(A, 0, 2, 1)
        "permute4" = Permute4([Id; 5]), // permute(A, 0, 2, 1, 3)
        "squeeze" = Squeeze([Id; 2]), // squeeze(A, axis)
        "unsqueeze" = Unsqueeze([Id; 2]), // unsqueeze(A, axis)
        "dummy" = Dummy,

        "sloop" = SLoop([Id; 5]), // Sequential loop => not used in rewriting phase
        "ploop" = PLoop([Id; 5]), // Parallel loop => not used in rewriting phase

        Num(i32),
        Var(egg::Symbol),
    }
}

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

#[derive(Default)]
pub struct LoopAnalysis;

#[derive(Debug, Clone, Default)]
pub struct LoopData {
    pub read_set: Vec<Access>,
    pub write_set: Vec<Access>,
    pub is_deleted: HashSet<TileLang>, // track deleted terms per eclass
    pub tensor_shape: Option<TensorShape>, // Shape information for tensors
}

// #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
// pub struct Access {
//     pub base: Option<String>,
//     pub index: Option<TileLang>,
// }
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Access {
    pub base: Option<String>,
    pub index: Vec<TileLang>, // Changed from Option<TileLang> to Vec<TileLang>
}

impl Analysis<TileLang> for LoopAnalysis {
    type Data = LoopData;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        let old_deleted_len = to.is_deleted.len();
        to.is_deleted.extend(from.is_deleted);
        to.read_set.extend(from.read_set);
        to.write_set.extend(from.write_set);

        // Merge tensor shapes - resolve wildcards if possible
        match (&to.tensor_shape, &from.tensor_shape) {
            (Some(to_shape), Some(from_shape)) => {
                // Check if shapes are compatible and merge wildcards
                if to_shape.dims.len() == from_shape.dims.len() {
                    let mut merged_dims = Vec::new();
                    let mut changed = false;

                    for (to_dim, from_dim) in to_shape.dims.iter().zip(&from_shape.dims) {
                        match (to_dim, from_dim) {
                            // Both concrete and equal - keep it
                            (Dimension::Concrete(t), Dimension::Concrete(_f)) => {
                                merged_dims.push(Dimension::Concrete(*t));
                            }
                            // Wildcard resolution - take concrete value
                            (Dimension::Wildcard, Dimension::Concrete(f)) => {
                                merged_dims.push(Dimension::Concrete(*f));
                                changed = true;
                            }
                            (Dimension::Concrete(t), Dimension::Wildcard) => {
                                merged_dims.push(Dimension::Concrete(*t));
                            }
                            // Both wildcards - keep wildcard
                            (Dimension::Wildcard, Dimension::Wildcard) => {
                                merged_dims.push(Dimension::Wildcard);
                            }
                        }
                    }

                    if changed {
                        to.tensor_shape = Some(TensorShape::new_with_dims(merged_dims));
                    }
                } else {
                    panic!("Mismatched tensor shapes during merge: different number of dimensions {} vs {}", 
                           to_shape.dims.len(), from_shape.dims.len());
                }
            }
            (None, Some(_)) => {
                to.tensor_shape = from.tensor_shape;
            }
            _ => {} // to has shape but from doesn't, or both are None - keep as is
        }

        DidMerge(
            // to.read_set.len() > old_read_len ||
            // to.write_set.len() > old_write_len ||
            // to.is_deleted.len() > old_deleted_len,
            // true,
            to.is_deleted.len() > old_deleted_len,
            true, // true, true
        )
    }

    fn make(egraph: &mut EGraph, enode: &TileLang) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;

        match enode {
            TileLang::Tile(_) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::ConstTile(_) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::FullTile => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Dummy => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Elem(_) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Const(_) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Input(_tensor) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Output(_tensor) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Tensor(_tensor) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Index(_args) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Load([base, idx]) => {
                // Get tensor shape from the shape tracker
                let tensor_shape = SHAPE_TRACKER.with(|tracker| {
                    let tracker = tracker.borrow();

                    // Get base tensor name
                    if let Some(base_name) = get_base_name_egraph(egraph, *base) {
                        // For comma-separated names like "A,B,C", use the first tensor's shape
                        let first_tensor_name =
                            base_name.split(',').next().unwrap_or(&base_name).trim();
                        if let Some(base_shape) = tracker.get_tensor_shape(first_tensor_name) {
                            // Extract dimensions from index pattern
                            let dims = extract_tile_dims(&tracker, egraph, idx, base_shape);
                            if !dims.is_empty() {
                                Some(TensorShape::new_with_dims(dims))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                });

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Store([_base, _val, _idx]) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Loop(_args) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::TLoop(_args) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::DLoop(_args) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::SLoop(_args) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::PLoop(_args) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Seq(_args) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Add([left, right]) => {
                // Element-wise operations - resolve wildcards if possible
                let tensor_shape = match (&x(left).tensor_shape, &x(right).tensor_shape) {
                    (Some(left_shape), Some(right_shape))
                        if left_shape.dims.len() == right_shape.dims.len() =>
                    {
                        let mut result_dims = Vec::new();
                        for (l_dim, r_dim) in left_shape.dims.iter().zip(&right_shape.dims) {
                            match (l_dim, r_dim) {
                                (Dimension::Concrete(l), Dimension::Concrete(r)) if l == r => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Concrete(r)) => {
                                    result_dims.push(Dimension::Concrete(*r));
                                }
                                (Dimension::Concrete(l), Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Wildcard);
                                }
                                _ => {
                                    return Self::Data {
                                        is_deleted: HashSet::new(),
                                        read_set: Vec::new(),
                                        write_set: Vec::new(),
                                        tensor_shape: None,
                                    }
                                }
                            }
                        }
                        Some(TensorShape::new_with_dims(result_dims))
                    }
                    (Some(shape), None) => Some(shape.clone()),
                    (None, Some(shape)) => Some(shape.clone()),
                    _ => None,
                };

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Le([left, right]) => {
                // Element-wise comparison - preserve input shape
                let tensor_shape = match (&x(left).tensor_shape, &x(right).tensor_shape) {
                    (Some(left_shape), Some(right_shape))
                        if left_shape.dims.len() == right_shape.dims.len() =>
                    {
                        let mut result_dims = Vec::new();
                        for (l_dim, r_dim) in left_shape.dims.iter().zip(&right_shape.dims) {
                            match (l_dim, r_dim) {
                                (Dimension::Concrete(l), Dimension::Concrete(r)) if l == r => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Concrete(r)) => {
                                    result_dims.push(Dimension::Concrete(*r));
                                }
                                (Dimension::Concrete(l), Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Wildcard);
                                }
                                _ => {
                                    return Self::Data {
                                        is_deleted: HashSet::new(),
                                        read_set: Vec::new(),
                                        write_set: Vec::new(),
                                        tensor_shape: None,
                                    }
                                }
                            }
                        }
                        Some(TensorShape::new_with_dims(result_dims))
                    }
                    (Some(shape), None) => Some(shape.clone()),
                    (None, Some(shape)) => Some(shape.clone()),
                    _ => None,
                };

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Max([left, right]) | TileLang::Min([left, right]) => {
                // Element-wise operations - resolve wildcards if possible
                let tensor_shape = match (&x(left).tensor_shape, &x(right).tensor_shape) {
                    (Some(left_shape), Some(right_shape))
                        if left_shape.dims.len() == right_shape.dims.len() =>
                    {
                        let mut result_dims = Vec::new();
                        for (l_dim, r_dim) in left_shape.dims.iter().zip(&right_shape.dims) {
                            match (l_dim, r_dim) {
                                (Dimension::Concrete(l), Dimension::Concrete(r)) if l == r => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Concrete(r)) => {
                                    result_dims.push(Dimension::Concrete(*r));
                                }
                                (Dimension::Concrete(l), Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Wildcard);
                                }
                                _ => {
                                    return Self::Data {
                                        is_deleted: HashSet::new(),
                                        read_set: Vec::new(),
                                        write_set: Vec::new(),
                                        tensor_shape: None,
                                    }
                                }
                            }
                        }
                        Some(TensorShape::new_with_dims(result_dims))
                    }
                    (Some(shape), None) => Some(shape.clone()),
                    (None, Some(shape)) => Some(shape.clone()),
                    _ => None,
                };

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Sub([left, right]) => {
                // Element-wise operations preserve the shape of inputs
                let tensor_shape = match (&x(left).tensor_shape, &x(right).tensor_shape) {
                    (Some(left_shape), Some(right_shape))
                        if left_shape.dims.len() == right_shape.dims.len() =>
                    {
                        let mut result_dims = Vec::new();
                        for (l_dim, r_dim) in left_shape.dims.iter().zip(&right_shape.dims) {
                            match (l_dim, r_dim) {
                                (Dimension::Concrete(l), Dimension::Concrete(r)) if l == r => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Concrete(r)) => {
                                    result_dims.push(Dimension::Concrete(*r));
                                }
                                (Dimension::Concrete(l), Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Wildcard);
                                }
                                _ => {
                                    return Self::Data {
                                        is_deleted: HashSet::new(),
                                        read_set: Vec::new(),
                                        write_set: Vec::new(),
                                        tensor_shape: None,
                                    }
                                }
                            }
                        }
                        Some(TensorShape::new_with_dims(result_dims))
                    }
                    (Some(shape), None) => Some(shape.clone()),
                    (None, Some(shape)) => Some(shape.clone()),
                    _ => None,
                };

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Div([left, right]) => {
                // Element-wise operations preserve the shape of inputs
                let tensor_shape = match (&x(left).tensor_shape, &x(right).tensor_shape) {
                    (Some(left_shape), Some(right_shape))
                        if left_shape.dims.len() == right_shape.dims.len() =>
                    {
                        let mut result_dims = Vec::new();
                        for (l_dim, r_dim) in left_shape.dims.iter().zip(&right_shape.dims) {
                            match (l_dim, r_dim) {
                                (Dimension::Concrete(l), Dimension::Concrete(r)) if l == r => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Concrete(r)) => {
                                    result_dims.push(Dimension::Concrete(*r));
                                }
                                (Dimension::Concrete(l), Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Wildcard);
                                }
                                _ => {
                                    return Self::Data {
                                        is_deleted: HashSet::new(),
                                        read_set: Vec::new(),
                                        write_set: Vec::new(),
                                        tensor_shape: None,
                                    }
                                }
                            }
                        }
                        Some(TensorShape::new_with_dims(result_dims))
                    }
                    (Some(shape), None) => Some(shape.clone()),
                    (None, Some(shape)) => Some(shape.clone()),
                    _ => None,
                };

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Mul([left, right]) => {
                // Element-wise operations preserve the shape of inputs
                let tensor_shape = match (&x(left).tensor_shape, &x(right).tensor_shape) {
                    (Some(left_shape), Some(right_shape))
                        if left_shape.dims.len() == right_shape.dims.len() =>
                    {
                        let mut result_dims = Vec::new();
                        for (l_dim, r_dim) in left_shape.dims.iter().zip(&right_shape.dims) {
                            match (l_dim, r_dim) {
                                (Dimension::Concrete(l), Dimension::Concrete(r)) if l == r => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Concrete(r)) => {
                                    result_dims.push(Dimension::Concrete(*r));
                                }
                                (Dimension::Concrete(l), Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Concrete(*l));
                                }
                                (Dimension::Wildcard, Dimension::Wildcard) => {
                                    result_dims.push(Dimension::Wildcard);
                                }
                                _ => {
                                    return Self::Data {
                                        is_deleted: HashSet::new(),
                                        read_set: Vec::new(),
                                        write_set: Vec::new(),
                                        tensor_shape: None,
                                    }
                                }
                            }
                        }
                        Some(TensorShape::new_with_dims(result_dims))
                    }
                    (Some(shape), None) => Some(shape.clone()),
                    (None, Some(shape)) => Some(shape.clone()),
                    _ => None,
                };

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Matmul([left, right]) => {
                // Matrix multiplication: (M, K) x (K, N) -> (M, N)
                let tensor_shape = match (&x(left).tensor_shape, &x(right).tensor_shape) {
                    (Some(left_shape), Some(right_shape)) => {
                        if left_shape.dims.len() >= 2 && right_shape.dims.len() >= 2 {
                            let mut result_dims = Vec::new();

                            // Handle batch dimensions - must match or be wildcard
                            if left_shape.dims.len() > 2 || right_shape.dims.len() > 2 {
                                let left_batch = &left_shape.dims[..left_shape.dims.len() - 2];
                                let right_batch = &right_shape.dims[..right_shape.dims.len() - 2];

                                // For simplicity, assume batch dims match in count
                                if left_batch.len() == right_batch.len() {
                                    for (l_dim, r_dim) in left_batch.iter().zip(right_batch) {
                                        match (l_dim, r_dim) {
                                            (Dimension::Concrete(l), Dimension::Concrete(r))
                                                if l == r =>
                                            {
                                                result_dims.push(Dimension::Concrete(*l));
                                            }
                                            (Dimension::Wildcard, Dimension::Concrete(r)) => {
                                                result_dims.push(Dimension::Concrete(*r));
                                            }
                                            (Dimension::Concrete(l), Dimension::Wildcard) => {
                                                result_dims.push(Dimension::Concrete(*l));
                                            }
                                            (Dimension::Wildcard, Dimension::Wildcard) => {
                                                result_dims.push(Dimension::Wildcard);
                                            }
                                            _ => {
                                                return Self::Data {
                                                    is_deleted: HashSet::new(),
                                                    read_set: Vec::new(),
                                                    write_set: Vec::new(),
                                                    tensor_shape: None,
                                                }
                                            }
                                        }
                                    }
                                } else if left_batch.len() > 0 {
                                    result_dims.extend_from_slice(left_batch);
                                } else if right_batch.len() > 0 {
                                    result_dims.extend_from_slice(right_batch);
                                }
                            }

                            // Add M dimension (from left)
                            result_dims.push(left_shape.dims[left_shape.dims.len() - 2].clone());

                            // Add N dimension (from right)
                            result_dims.push(right_shape.dims[right_shape.dims.len() - 1].clone());

                            Some(TensorShape::new_with_dims(result_dims))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Concat([left, right, axis]) => {
                // Concat combines two tensors along specified axis
                let tensor_shape = match (&x(left).tensor_shape, &x(right).tensor_shape) {
                    (Some(left_shape), Some(right_shape)) => {
                        // Get the axis value if it's a constant
                        egraph[*axis].nodes.iter().find_map(|n| match n {
                            TileLang::Num(axis_val) => {
                                let axis_idx = *axis_val as usize;
                                if axis_idx < left_shape.dims.len()
                                    && left_shape.dims.len() == right_shape.dims.len()
                                {
                                    let mut result_dims = Vec::new();

                                    // Check all dimensions
                                    for i in 0..left_shape.dims.len() {
                                        if i == axis_idx {
                                            // Concatenation axis - add the dimensions
                                            match (&left_shape.dims[i], &right_shape.dims[i]) {
                                                (
                                                    Dimension::Concrete(l),
                                                    Dimension::Concrete(r),
                                                ) => {
                                                    result_dims.push(Dimension::Concrete(l + r));
                                                }
                                                (Dimension::Wildcard, _)
                                                | (_, Dimension::Wildcard) => {
                                                    // If either is wildcard, result is wildcard
                                                    result_dims.push(Dimension::Wildcard);
                                                }
                                            }
                                        } else {
                                            // Non-concat dimensions must match
                                            match (&left_shape.dims[i], &right_shape.dims[i]) {
                                                (
                                                    Dimension::Concrete(l),
                                                    Dimension::Concrete(r),
                                                ) if l == r => {
                                                    result_dims.push(Dimension::Concrete(*l));
                                                }
                                                (Dimension::Wildcard, Dimension::Concrete(r)) => {
                                                    result_dims.push(Dimension::Concrete(*r));
                                                }
                                                (Dimension::Concrete(l), Dimension::Wildcard) => {
                                                    result_dims.push(Dimension::Concrete(*l));
                                                }
                                                (Dimension::Wildcard, Dimension::Wildcard) => {
                                                    result_dims.push(Dimension::Wildcard);
                                                }
                                                _ => return None, // Dimension mismatch
                                            }
                                        }
                                    }

                                    Some(TensorShape::new_with_dims(result_dims))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        })
                    }
                    _ => None,
                };

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Exp(arg) => {
                // Exp is element-wise, preserves input shape
                let tensor_shape = x(arg).tensor_shape.clone();

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Sqr(arg) => {
                // Exp is element-wise, preserves input shape
                let tensor_shape = x(arg).tensor_shape.clone();

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Sqrt(arg) => {
                // Exp is element-wise, preserves input shape
                let tensor_shape = x(arg).tensor_shape.clone();

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Sigmoid(arg) => {
                // Sigmoid is element-wise, preserves input shape
                let tensor_shape = x(arg).tensor_shape.clone();

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Erf(arg) => {
                // Erf is element-wise, preserves input shape
                let tensor_shape = x(arg).tensor_shape.clone();

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Abs(arg) => {
                // Abs is element-wise, preserves input shape
                let tensor_shape = x(arg).tensor_shape.clone();

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Cast([_dtype, arg]) => {
                // Cast preserves input shape
                let tensor_shape = x(arg).tensor_shape.clone();

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::ReduceSum([input, axis]) => {
                // ReduceSum removes the specified axis dimension
                let tensor_shape = x(input).tensor_shape.as_ref().and_then(|input_shape| {
                    // Get the axis value if it's a constant
                    egraph[*axis].nodes.iter().find_map(|n| match n {
                        TileLang::Num(axis_val) => {
                            let axis_idx = *axis_val as usize;
                            if axis_idx < input_shape.dims.len() {
                                let mut result_dims = input_shape.dims.clone();
                                result_dims.remove(axis_idx);
                                Some(TensorShape::new_with_dims(result_dims))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                });

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::ReduceMin([input, axis]) | TileLang::ReduceMax([input, axis]) => {
                // ReduceMin/ReduceMax removes the specified axis dimension
                let tensor_shape = x(input).tensor_shape.as_ref().and_then(|input_shape| {
                    egraph[*axis].nodes.iter().find_map(|n| match n {
                        TileLang::Num(axis_val) => {
                            let axis_idx = *axis_val as usize;
                            if axis_idx < input_shape.dims.len() {
                                let mut result_dims = input_shape.dims.clone();
                                result_dims.remove(axis_idx);
                                Some(TensorShape::new_with_dims(result_dims))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                });

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Broadcast([input, axis]) => {
                // Broadcast adds a wildcard dimension at the specified axis
                let tensor_shape = x(input).tensor_shape.as_ref().and_then(|input_shape| {
                    egraph[*axis].nodes.iter().find_map(|n| match n {
                        TileLang::Num(axis_val) => {
                            let axis_idx = *axis_val as usize;
                            if axis_idx <= input_shape.dims.len() {
                                let mut result_dims = input_shape.dims.clone();
                                // Insert a wildcard dimension at the specified axis
                                result_dims.insert(axis_idx, Dimension::Wildcard);
                                Some(TensorShape::new_with_dims(result_dims))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                });

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Transpose(input) => {
                // Transpose swaps the last two dimensions (e.g., MxN -> NxM).
                // If rank < 2, shape inference is not possible here.
                let tensor_shape = x(input).tensor_shape.as_ref().and_then(|input_shape| {
                    let rank = input_shape.dims.len();
                    if rank >= 2 {
                        let mut result_dims = input_shape.dims.clone();
                        result_dims.swap(rank - 2, rank - 1);
                        Some(TensorShape::new_with_dims(result_dims))
                    } else {
                        None
                    }
                });

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Permute3([input, axis0, axis1, axis2]) => {
                // Permute reorders dimensions according to specified axes
                let tensor_shape = x(input).tensor_shape.as_ref().and_then(|input_shape| {
                    if input_shape.dims.len() >= 3 {
                        // Get axis values if they're constants
                        let axes: Vec<Option<usize>> = vec![axis0, axis1, axis2]
                            .into_iter()
                            .map(|axis_id| {
                                egraph[*axis_id].nodes.iter().find_map(|n| match n {
                                    TileLang::Num(val) => Some(*val as usize),
                                    _ => None,
                                })
                            })
                            .collect();

                        if axes.iter().all(|a| a.is_some()) {
                            let mut result_dims = vec![Dimension::Concrete(0); 3];
                            for i in 0..3 {
                                if let Some(src_idx) = axes[i] {
                                    if src_idx < input_shape.dims.len() {
                                        result_dims[i] = input_shape.dims[src_idx].clone();
                                    }
                                }
                            }
                            Some(TensorShape::new_with_dims(result_dims))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                });

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Permute4([input, axis0, axis1, axis2, axis3]) => {
                // Permute reorders 4D dimensions according to specified axes
                let tensor_shape = x(input).tensor_shape.as_ref().and_then(|input_shape| {
                    if input_shape.dims.len() >= 4 {
                        let axes: Vec<Option<usize>> = vec![axis0, axis1, axis2, axis3]
                            .into_iter()
                            .map(|axis_id| {
                                egraph[*axis_id].nodes.iter().find_map(|n| match n {
                                    TileLang::Num(val) => Some(*val as usize),
                                    _ => None,
                                })
                            })
                            .collect();

                        if axes.iter().all(|a| a.is_some()) {
                            let mut result_dims = vec![Dimension::Concrete(0); 4];
                            for i in 0..4 {
                                if let Some(src_idx) = axes[i] {
                                    if src_idx < input_shape.dims.len() {
                                        result_dims[i] = input_shape.dims[src_idx].clone();
                                    }
                                }
                            }
                            Some(TensorShape::new_with_dims(result_dims))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                });

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Squeeze([input, axis]) => {
                // Squeeze removes a dimension of size 1 at the specified axis
                let tensor_shape = x(input).tensor_shape.as_ref().and_then(|input_shape| {
                    egraph[*axis].nodes.iter().find_map(|n| match n {
                        TileLang::Num(axis_val) => {
                            let axis_idx = *axis_val as usize;
                            if axis_idx < input_shape.dims.len() {
                                // Check if the dimension at axis_idx is 1 (or wildcard)
                                match &input_shape.dims[axis_idx] {
                                    Dimension::Concrete(1) | Dimension::Wildcard => {
                                        let mut result_dims = input_shape.dims.clone();
                                        result_dims.remove(axis_idx);
                                        Some(TensorShape::new_with_dims(result_dims))
                                    }
                                    _ => None, // Can't squeeze non-1 dimension
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                });

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Unsqueeze([input, axis]) => {
                // Unsqueeze adds a dimension of size 1 at the specified axis
                let tensor_shape = x(input).tensor_shape.as_ref().and_then(|input_shape| {
                    egraph[*axis].nodes.iter().find_map(|n| match n {
                        TileLang::Num(axis_val) => {
                            let axis_idx = *axis_val as usize;
                            if axis_idx <= input_shape.dims.len() {
                                let mut result_dims = input_shape.dims.clone();
                                result_dims.insert(axis_idx, Dimension::Concrete(1));
                                Some(TensorShape::new_with_dims(result_dims))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                });

                Self::Data {
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                    tensor_shape,
                }
            }
            TileLang::Var(_) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
            TileLang::Num(_) => Self::Data {
                is_deleted: HashSet::new(),
                read_set: Vec::new(),
                write_set: Vec::new(),
                tensor_shape: None,
            },
        }
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        // ====================================
        // (1) Sequence flattening
        // ====================================
        let nodes = egraph[id].nodes.clone();
        for node in nodes {
            if egraph[id].data.is_deleted.contains(&node) {
                continue;
            }

            if is_legal_seq_node(egraph, &node) {
                continue; // there already exists legal sequence node
            }

            let mut new_forms = vec![];
            if let TileLang::Seq([left, right]) = node {
                // let all_sequences = flatten_seq_all_branch(egraph, id, false);
                // for seq_elements in all_sequences {
                //     if seq_elements.len() > 1 {
                //         let mut iter = seq_elements.into_iter().rev();
                //         let mut current = iter.next().unwrap();
                //         while let Some(prev) = iter.next() {
                //             current = egraph.add(TileLang::Seq([prev, current]));
                //         }
                //         new_forms.push(current);
                //     }
                // }
                // for form in new_forms {
                //     egraph.union(id, form);
                // }
                // // egraph[id].data.is_deleted.insert(node);

                // Step 1: Flatten recursively
                let mut seq_elements = vec![];
                flatten_seq(egraph, left, &mut seq_elements);
                flatten_seq(egraph, right, &mut seq_elements);

                if seq_elements.is_empty() {
                    continue;
                }

                // Step 2: Rebuild canonical nested structure
                let mut iter = seq_elements.into_iter().rev();

                let mut current = iter.next().unwrap(); // last element (no seq_end)

                while let Some(prev) = iter.next() {
                    current = egraph.add(TileLang::Seq([prev, current]));
                }
                new_forms.push(current);

                for form in new_forms {
                    egraph.union(id, form);
                }
            }
        }

        // ====================================
        // (2) Value Forwarding
        // ====================================
        let nodes = egraph[id].nodes.clone();
        for node in nodes {
            if egraph[id].data.is_deleted.contains(&node) {
                continue;
            }

            if !is_legal_seq_node(egraph, &node) {
                continue;
            }
            let TileLang::Seq([left, right]) = node else {
                continue; // If not a Seq node, skip
            };

            // if contains_loop(egraph, left) || is_reduction_op(egraph, left){
            //     continue; // If the first op is loop, skip
            // }
            if is_reduction_op(egraph, left) {
                continue;
            }
            // flatten the sequences
            let mut seq_elements = vec![];
            flatten_seq(egraph, left, &mut seq_elements);
            flatten_seq(egraph, right, &mut seq_elements);

            if seq_elements.len() < 2 {
                continue;
            }

            // println!("            try memory forwarding at {:?}", seq_elements);

            let mut iter = seq_elements.into_iter();
            let op1 = iter.next().unwrap();

            // op1 must be (store A, val, idx)
            let Some((base_name, idx_expr, val_id)) = extract_store_info(egraph, op1) else {
                continue;
            };

            // println!("{:?}", base_name);
            // println!("{:?}", idx_expr);

            for op_i in iter {
                // if op_i is loop or reduction op, skip
                if contains_loop(egraph, op_i) {
                    continue;
                }

                // If op_i.readset.base != basename, skip
                let (op_read_set, _) = collect_access_sets(egraph, op_i, false);
                let dependent = op_read_set
                    .iter()
                    .any(|access| access.base.as_ref() == Some(&base_name));
                if !dependent {
                    // println!("                 No dependency");
                    continue;
                }

                // println!("            try memory forwarding at {:?}", op_i);

                // Walk subgraph rooted at op_i to find all reachable e-classes
                let subgraph = collect_reachable_eclasses(egraph, op_i);

                // Find all (load A idx) eclass from the subgraph
                let mut load_targets = vec![];
                for &eclass_id in &subgraph {
                    let data = &egraph[eclass_id].data;
                    for enode in &egraph[eclass_id].nodes {
                        if data.is_deleted.contains(&enode) {
                            continue;
                        }
                        if let TileLang::Load([base_id, load_idx_id]) = enode {
                            let base_match = egraph[*base_id].nodes.iter().any(|n| {
                                match n {
                                    TileLang::Input(tensor_id) | TileLang::Output(tensor_id) | TileLang::Tensor(tensor_id)=> {
                                        egraph[*tensor_id].nodes.iter().any(|tn| {
                                            matches!(tn, TileLang::Var(sym) if sym.as_str() == base_name)
                                        })
                                    }
                                    _ => false,
                                }
                            });

                            let idx_match = extract_expr(egraph, *load_idx_id) == idx_expr;
                            // let idx_match = extract_expr(egraph, *load_idx_id).as_ref() == Some(&idx_expr);

                            // println!("            find load node {:?}, {:?}", base_id, load_idx_id);
                            // println!("            base match: {:?}, idx match: {:?}", base_match, idx_match);

                            if base_match && idx_match {
                                load_targets.push(eclass_id);
                            }
                        }
                    }
                }

                // println!("            find load target {:?}", load_targets);

                // Replace all parent nodes that use the load with the value
                for &eclass_id in &subgraph {
                    let enodes = egraph[eclass_id].nodes.clone();
                    for enode in enodes {
                        if egraph[eclass_id].data.is_deleted.contains(&enode) {
                            continue;
                        }
                        let mut new_children = enode.children().to_vec();
                        let mut changed = false;

                        for i in 0..new_children.len() {
                            if load_targets.contains(&new_children[i]) {
                                new_children[i] = val_id;
                                changed = true;
                            }
                        }

                        if changed {
                            if let Ok(new_node) =
                                TileLang::from_op(&enode.to_string(), new_children)
                            {
                                let new_id = egraph.add(new_node);
                                egraph.union(eclass_id, new_id); // ✅ merge into same eclass
                                                                 // print_eclass(egraph, val_id);
                                                                 // println!("{:?}", extract_expr(egraph, val_id).as_ref());
                                                                 // egraph[eclass_id].data.is_deleted.insert(enode.clone()); // ✅ Mark the old enode as deleted
                            }
                        }
                    }
                }
            }
        }

        // ====================================
        // (3) Resolve tmp_loop
        // ====================================
        let nodes = egraph[id].nodes.clone();
        for node in nodes {
            // if egraph[id].data.is_deleted.contains(&node) {
            //     continue;
            // }

            if let TileLang::TLoop([start, end, tile, loop_var1_id, loop_var2_id, body_id]) = node {
                let has_loop_already = egraph[id]
                    .nodes
                    .iter()
                    .any(|n| matches!(n, TileLang::Loop([s, e, t, lv, b])
                        if *s == start && *e == end && *t == tile && *lv == loop_var1_id && *b == body_id
                    ));

                if !has_loop_already {
                    // Insert new Loop(...) node into the current eclass
                    let loop_node = TileLang::Loop([start, end, tile, loop_var1_id, body_id]);
                    let loop_id = egraph.add(loop_node);
                    egraph.union(id, loop_id);
                } else {
                    continue;
                }

                let reachable = collect_reachable_eclasses(egraph, body_id);

                for &eclass_id in &reachable {
                    let enodes = egraph[eclass_id].nodes.clone();
                    for enode in enodes {
                        if let TileLang::Tile(arg_id) = enode {
                            let loop_var2_sym = get_var_symbol(egraph, loop_var2_id);

                            let is_loop_var2 =
                                egraph[arg_id]
                                    .nodes
                                    .iter()
                                    .any(|n| match (n, &loop_var2_sym) {
                                        (TileLang::Var(sym), Some(expected)) => sym == *expected,
                                        _ => false,
                                    });
                            if is_loop_var2 {
                                let Some(loop_var1_sym) = get_var_symbol(egraph, loop_var1_id)
                                else {
                                    continue;
                                };
                                let loop_var1_node = TileLang::Var(loop_var1_sym.clone());
                                let loop_var1_eclass = egraph.add(loop_var1_node);

                                let elem_node = TileLang::Elem(loop_var1_eclass);
                                let elem_id = egraph.add(elem_node);

                                egraph.union(eclass_id, elem_id);
                            }
                        }
                    }
                }

                // egraph[id].data.is_deleted.insert(node);
            }
        }
    }
}

fn extract_tile_dims(
    tracker: &ShapeTracker,
    egraph: &EGraph,
    idx_id: &Id,
    base_shape: &TensorShape,
) -> Vec<Dimension> {
    extract_tile_dims_helper(tracker, egraph, idx_id, base_shape, &mut 0)
}

fn extract_tile_dims_helper(
    tracker: &ShapeTracker,
    egraph: &EGraph,
    idx_id: &Id,
    base_shape: &TensorShape,
    current_dim: &mut usize,
) -> Vec<Dimension> {
    let mut dims = Vec::new();

    if let Some(index_node) = egraph[*idx_id]
        .nodes
        .iter()
        .find(|n| matches!(n, TileLang::Index(_)))
    {
        if let TileLang::Index(tile_indices) = index_node {
            for tile_id in tile_indices.iter() {
                if *current_dim < base_shape.dims.len() {
                    let base_dim = &base_shape.dims[*current_dim];

                    if egraph[*tile_id]
                        .nodes
                        .iter()
                        .any(|n| matches!(n, TileLang::Index(_)))
                    {
                        // Nested index - recursively process without incrementing current_dim yet
                        let nested_dims = extract_tile_dims_helper(
                            tracker,
                            egraph,
                            tile_id,
                            base_shape,
                            current_dim,
                        );
                        dims.extend(nested_dims);
                    } else {
                        // Regular tile - process and increment current_dim
                        match base_dim {
                            Dimension::Concrete(full_dim) => {
                                // Check if there are multiple Tile or Elem nodes with different loop variables
                                let tile_elem_nodes: Vec<_> = egraph[*tile_id]
                                    .nodes
                                    .iter()
                                    .filter(|n| matches!(n, TileLang::Tile(_) | TileLang::Elem(_)))
                                    .collect();

                                let tile_dim = if tile_elem_nodes.len() > 1 {
                                    // Multiple Tile/Elem nodes - check if they have different loop variables
                                    let mut loop_vars = std::collections::HashSet::new();
                                    for node in &tile_elem_nodes {
                                        match node {
                                            TileLang::Tile(var_id) | TileLang::Elem(var_id) => {
                                                loop_vars.insert(var_id);
                                            }
                                            _ => {}
                                        }
                                    }

                                    if loop_vars.len() > 1 {
                                        // Different loop variables detected - use Wildcard
                                        Some(Dimension::Wildcard)
                                    } else {
                                        // Same loop variable - proceed normally
                                        egraph[*tile_id].nodes.iter().find_map(|n| match n {
                                            TileLang::Tile(_) => Some(Dimension::Concrete(
                                                tracker.get_tile_size(*full_dim),
                                            )),
                                            TileLang::Elem(_) => Some(Dimension::Concrete(1)),
                                            _ => None,
                                        })
                                    }
                                } else {
                                    // Single node or no Tile/Elem nodes - proceed with normal logic
                                    egraph[*tile_id].nodes.iter().find_map(|n| match n {
                                        TileLang::Tile(_) => Some(Dimension::Concrete(
                                            tracker.get_tile_size(*full_dim),
                                        )),
                                        TileLang::FullTile => Some(Dimension::Concrete(*full_dim)),
                                        TileLang::ConstTile([_, interval_id]) => {
                                            egraph[*interval_id].nodes.iter().find_map(
                                                |interval_node| match interval_node {
                                                    TileLang::Num(n) => {
                                                        Some(Dimension::Concrete(*n as usize))
                                                    }
                                                    _ => None,
                                                },
                                            )
                                        }
                                        TileLang::Elem(_) => Some(Dimension::Concrete(1)),
                                        _ => None,
                                    })
                                };

                                if let Some(dim) = tile_dim {
                                    dims.push(dim);
                                }
                            }
                            Dimension::Wildcard => {
                                dims.push(Dimension::Wildcard);
                            }
                        }
                        *current_dim += 1;
                    }
                }
            }
        }
    }

    dims
}
