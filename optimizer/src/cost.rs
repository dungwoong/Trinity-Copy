use crate::language::*;
use crate::shape::{Dimension, TensorShape};
use crate::utils::*;
use egg::*;

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

// Fine-grained cost model
pub struct FineGrainedCost<'a> {
    pub egraph: &'a EGraph,
    pub cost_model: FineGrainedCostModel,
}

impl CostFunction<TileLang> for FineGrainedCost<'_> {
    type Cost = u64;
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &TileLang, mut costs: C) -> Self::Cost {
        let self_cost = self.cost_model.get_flops(self.egraph, enode);

        // Check if this node itself is a dummy variable
        let is_dummy_var = match enode {
            TileLang::Var(sym) => sym.as_str().starts_with("dummydata"),
            _ => false,
        };

        // If self_cost is already MAX
        if is_dummy_var {
            return 2000000000000;
        }

        // Accumulate child costs
        let total = enode.fold(self_cost, |sum, id| {
            let child_cost = costs(id);
            if child_cost == 2000000000000 {
                return 2000000000000;
            } else {
                // sum.saturating_add(child_cost).saturating_sub(1)
                sum + child_cost
            }
        });

        // If total is MAX but this isn't a dummy var, return MAX - 1
        if total == 2000000000000 && !is_dummy_var {
            2000000000000
        } else {
            total
        }
    }
}

pub struct FineGrainedCostModel {
    pub tile_size: u64,
}

impl FineGrainedCostModel {
    pub fn new() -> Self {
        FineGrainedCostModel { tile_size: 64 }
    }
    pub fn with_tile_size(tile_size: u64) -> Self {
        FineGrainedCostModel { tile_size }
    }
    pub fn get_flops(&self, egraph: &EGraph, enode: &TileLang) -> u64 {
        match enode {
            TileLang::Var(sym) => {
                // Give maximum cost to dummy variables
                if sym.as_str().starts_with("dummydata") {
                    2000000000000
                } else {
                    0
                }
            }

            TileLang::Matmul([left, right]) => {
                // Scale down by 1000 to prevent overflow
                let cost = self.get_matmul_flops(egraph, *left, *right);
                // if cost != 0 {
                //     println!("cost: {:?}", cost);
                // }
                cost
            }

            TileLang::Add([left, right])
            | TileLang::Sub([left, right])
            | TileLang::Mul([left, right])
            | TileLang::Div([left, right])
            | TileLang::Le([left, right])
            | TileLang::Max([left, right])
            | TileLang::Min([left, right]) => {
                // Scale down by 1000 to prevent overflow
                let cost = self.get_elementwise_flops(egraph, *left, *right);
                // if cost != 0 {
                //     println!("cost: {:?}", cost);
                // }
                cost
            }

            TileLang::Sqr(input)
            | TileLang::Sqrt(input)
            | TileLang::Exp(input)
            | TileLang::Sigmoid(input)
            | TileLang::Erf(input)
            | TileLang::Abs(input)
            | TileLang::Cast([_, input]) => {
                // Scale down by 1000 to prevent overflow
                let cost = self.get_unary_flops(egraph, *input, 10);
                // if cost != 0 {
                //     println!("cost: {:?}", cost);
                // }
                cost
            }

            TileLang::ReduceSum([input, _axis])
            | TileLang::ReduceMin([input, _axis])
            | TileLang::ReduceMax([input, _axis]) => {
                // Scale down by 1000 to prevent overflow
                self.get_reduce_flops(egraph, *input)
            }

            _ => 0,
        }
    }

    fn get_matmul_flops(&self, egraph: &EGraph, left_id: Id, right_id: Id) -> u64 {
        let left_shape = self.get_tensor_shape(egraph, left_id);
        let right_shape = self.get_tensor_shape(egraph, right_id);
        match (left_shape, right_shape) {
            (Some(left), Some(right)) => {
                if left.dims.len() >= 2 && right.dims.len() >= 2 {
                    let batch_size = if left.dims.len() > 2 {
                        self.calculate_batch_size(&left.dims[..left.dims.len() - 2])
                    } else {
                        1
                    };

                    let (m, k) = match (
                        &left.dims[left.dims.len() - 2],
                        &left.dims[left.dims.len() - 1],
                    ) {
                        (Dimension::Concrete(m), Dimension::Concrete(k)) => (*m as u64, *k as u64),
                        _ => return 0,
                    };

                    let n = match &right.dims[right.dims.len() - 1] {
                        Dimension::Concrete(n) => *n as u64,
                        _ => return 0,
                    };

                    batch_size * m * n * (2 * k - 1)
                } else {
                    0
                }
            }
            (None, None) => {
                println!("Shape is unknown");
                0
            }
            _ => 0,
        }
    }

    fn get_elementwise_flops(&self, egraph: &EGraph, left_id: Id, right_id: Id) -> u64 {
        let shape = match (
            self.get_tensor_shape(egraph, left_id),
            self.get_tensor_shape(egraph, right_id),
        ) {
            (Some(left), Some(right)) => {
                if self.shapes_match(&left, &right) {
                    Some(left)
                } else {
                    None
                }
            }
            (Some(shape), None) | (None, Some(shape)) => Some(shape),
            _ => {
                // println!("Shape is unknown!");
                None
            }
        };

        match shape {
            Some(s) => self.calculate_total_elements(&s),
            None => {
                // println!("Shape is none");
                0
            }
        }
    }

    fn get_unary_flops(&self, egraph: &EGraph, input_id: Id, flops_per_element: u64) -> u64 {
        match self.get_tensor_shape(egraph, input_id) {
            Some(shape) => self.calculate_total_elements(&shape) * flops_per_element,
            None => {
                // println!("Shape is none");
                0
            }
        }
    }

    fn get_reduce_flops(&self, egraph: &EGraph, input_id: Id) -> u64 {
        match self.get_tensor_shape(egraph, input_id) {
            Some(shape) => self.calculate_total_elements(&shape),
            None => {
                // println!("Shape is none");
                0
            }
        }
    }

    fn get_tensor_shape(&self, egraph: &EGraph, id: Id) -> Option<TensorShape> {
        egraph[id].data.tensor_shape.clone()
    }
    fn shapes_match(&self, left: &TensorShape, right: &TensorShape) -> bool {
        if left.dims.len() != right.dims.len() {
            return false;
        }
        left.dims
            .iter()
            .zip(&right.dims)
            .all(|(l, r)| match (l, r) {
                (Dimension::Concrete(l_size), Dimension::Concrete(r_size)) => l_size == r_size,
                (Dimension::Wildcard, _) | (_, Dimension::Wildcard) => true,
            })
    }
    fn calculate_total_elements(&self, shape: &TensorShape) -> u64 {
        shape.dims.iter().fold(1u64, |acc, dim| match dim {
            Dimension::Concrete(size) => acc * (*size as u64),
            Dimension::Wildcard => acc,
        })
    }
    fn calculate_batch_size(&self, batch_dims: &[Dimension]) -> u64 {
        batch_dims.iter().fold(1u64, |acc, dim| match dim {
            Dimension::Concrete(size) => acc * (*size as u64),
            Dimension::Wildcard => acc,
        })
    }
}

pub fn create_fine_grained_extractor<'a>(
    egraph: &'a EGraph,
) -> Extractor<'a, FineGrainedCost<'a>, TileLang, LoopAnalysis> {
    let cost_model = FineGrainedCostModel::new();

    let cost_function = FineGrainedCost { egraph, cost_model };

    Extractor::new(egraph, cost_function)
}

pub fn create_fine_grained_extractor_with_tile_size<'a>(
    egraph: &'a EGraph,
    tile_size: u64,
) -> Extractor<'a, FineGrainedCost<'a>, TileLang, LoopAnalysis> {
    let cost_model = FineGrainedCostModel::with_tile_size(tile_size);

    let cost_function = FineGrainedCost { egraph, cost_model };

    Extractor::new(egraph, cost_function)
}

// Wrapper class for egg's cost function
pub struct TileCost<'a> {
    pub egraph: &'a EGraph,
    pub cost_model: TileCostModel,
    // pub cost_model: &'a TileCostModel,
}

impl CostFunction<TileLang> for TileCost<'_> {
    type Cost = usize;
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &TileLang, mut costs: C) -> Self::Cost {
        let self_cost = self.cost_model.get_self_cost(self.egraph, enode);
        enode.fold(self_cost, |sum, id| sum + costs(id))
    }
}

// Class for our cost model - counts loops with cross-iteration dependencies
pub struct TileCostModel;

impl TileCostModel {
    pub fn new() -> Self {
        TileCostModel
    }

    pub fn get_self_cost(&self, egraph: &EGraph, enode: &TileLang) -> usize {
        match enode {
            TileLang::Loop([_start, _end, _tile, loop_var, body]) => {
                if has_loop_carried_dependency_egraph(egraph, *body, *loop_var) {
                    1
                } else {
                    1
                }
            }
            _ => 0,
        }
    }

    pub fn get_self_cost_should_sequential(
        &self,
        egraph: &EGraph,
        enode: &TileLang,
        should_sequential: bool,
    ) -> usize {
        match enode {
            TileLang::Loop([_start, _end, _tile, loop_var, body]) => {
                if has_loop_carried_dependency_egraph(egraph, *body, *loop_var) || should_sequential
                {
                    1
                } else {
                    0
                }
            }
            _ => 0,
        }
    }
}

pub fn create_extractor<'a>(
    egraph: &'a EGraph,
) -> Extractor<'a, TileCost<'a>, TileLang, LoopAnalysis> {
    // Create simple cost model
    let cost_model = TileCostModel::new();

    // Create cost function wrapper
    let cost_function = TileCost { egraph, cost_model };

    // Create and return extractor
    Extractor::new(egraph, cost_function)
}
