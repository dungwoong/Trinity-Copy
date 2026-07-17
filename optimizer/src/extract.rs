//! Extracting high performance potential expressions

use crate::language::*;
use egg::*;
use num_bigint::BigUint;
use std::collections::HashMap;
use std::collections::HashSet;

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;
const MAX_DEPTH: usize = 100;

// Strawman approach: Enumerate all possible expressions
pub fn enumerate_expressions_all(egraph: &EGraph, eclass_id: Id) -> Vec<String> {
    let mut visited = HashSet::new();
    let extractor = Extractor::new(egraph, AstSize);
    enumerate_recursive_with_parent(egraph, eclass_id, &mut visited, 0, None, 0, &extractor)
}

pub fn enumerate_recursive_with_parent(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    depth: usize,
    parent_node: Option<&TileLang>,
    child_index: usize,
    extractor: &Extractor<AstSize, TileLang, LoopAnalysis>,
) -> Vec<String> {
    // Constraint 2: Handle cycles - allow only once
    if visited.contains(&eclass_id) {
        let best_expr = extractor.find_best(eclass_id);
        return vec![format!("{}", best_expr.1)]; // .1 is the extracted expression
    }
    // Depth limit (separate from cycle detection)
    if depth > MAX_DEPTH {
        return vec![format!("depth_limit")];
    }

    visited.insert(eclass_id);
    let mut results = Vec::new();
    let eclass = &egraph[eclass_id];

    for enode in &eclass.nodes {
        // FILTER: Skip Seq nodes that are left children of parent Seq nodes
        if let TileLang::Seq(_) = enode {
            if let Some(TileLang::Seq(_)) = parent_node {
                if child_index == 0 {
                    // Left child (index 0)
                    // println!("Skipping Seq as left child of parent Seq at depth: {:?}", depth);
                    continue;
                }
            }
        }

        if let TileLang::TLoop(_) = enode {
            continue;
        }

        if should_skip_seq_for_commutativity(egraph, eclass_id, enode) {
            continue;
        }

        let children = enode.children();

        if children.is_empty() {
            // Leaf node
            results.push(format!("{}", enode));
        } else {
            // Get expressions for each child
            let mut child_expressions = Vec::new();
            for (index, &child_id) in children.iter().enumerate() {
                let child_exprs = enumerate_recursive_with_parent(
                    egraph,
                    child_id,
                    visited,
                    depth + 1,
                    Some(enode), // Pass current node as parent
                    index,       // Pass child index
                    extractor,
                );
                child_expressions.push(child_exprs);
            }

            // Generate cartesian product
            let combinations = cartesian_product(&child_expressions);
            for combo in combinations {
                let expr_str = format_enode_with_children(enode, &combo);
                results.push(expr_str);
            }
        }
    }

    visited.remove(&eclass_id);
    results
}

// Heuristic approach: extract expression with minimal kernels
#[derive(Debug, Clone)]
pub struct SemiExpression {
    structure: String,
    kernel_count: usize,
    path: StructuralPath, // Track the specific path through the egraph that led to this semi-expression
}

#[derive(Debug, Clone)]
pub struct StructuralPath {
    root_eclass: Id,
    node_choices: HashMap<Id, TileLang>,
    traversal_order: Vec<Id>,
}

pub fn enumerate_expressions_num_kernel(
    egraph: &EGraph,
    eclass_id: Id,
    top_percentage: f32,
) -> Vec<String> {
    // Step1: Extract semi-expressions to calculate the number of gpu kernel
    let semi_expressions = enumerate_semi_expressions(egraph, eclass_id);

    // Step2: Sort by kerne count and pick top n%
    let mut sorted_semi = semi_expressions;
    sorted_semi.sort_by_key(|semi| semi.kernel_count);

    let top_count = ((sorted_semi.len() as f32) * top_percentage).max(1.0) as usize;
    let selected_semi = &sorted_semi[..top_count.min(sorted_semi.len())];

    println!(
        "Selected {} semi-expressions out of {} (top {:.1}%)",
        selected_semi.len(),
        sorted_semi.len(),
        top_percentage * 100.0
    );

    // Step 3: Enumerate full expressions only for selected path
    let mut all_results = Vec::new();
    for semi in selected_semi {
        let full_expressions = enumerate_full_expressions_from_semi(egraph, &semi.path);
        all_results.extend(full_expressions);
    }

    all_results
}

pub fn enumerate_semi_expressions(egraph: &EGraph, eclass_id: Id) -> Vec<SemiExpression> {
    let mut visited = HashSet::new();
    let mut path = StructuralPath {
        root_eclass: eclass_id,
        node_choices: HashMap::new(),
        traversal_order: Vec::new(),
    };

    enumerate_semi_recursive_with_path(egraph, eclass_id, &mut visited, 0, &mut path, None, 0)
}

pub fn enumerate_semi_recursive_with_path(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    loop_level: usize,
    current_path: &mut StructuralPath,
    parent_node: Option<&TileLang>,
    child_index: usize,
) -> Vec<SemiExpression> {
    // Handle cycles - allow only once
    if visited.contains(&eclass_id) {
        return vec![SemiExpression {
            structure: "(cycle)".to_string(),
            kernel_count: 0,
            path: current_path.clone(),
        }];
    }

    visited.insert(eclass_id);
    let mut results = Vec::new();
    let eclass = &egraph[eclass_id];

    current_path.traversal_order.push(eclass_id);

    for enode in &eclass.nodes {
        // FILTER: Skip Seq nodes that are left children of parent Seq nodes
        if let TileLang::Seq(_) = enode {
            if let Some(TileLang::Seq(_)) = parent_node {
                if child_index == 0 {
                    continue;
                }
            }
        }

        // FILTER: TLoop is not a valid operator
        if let TileLang::TLoop(_) = enode {
            continue;
        }

        if should_skip_seq_for_commutativity(egraph, eclass_id, enode) {
            continue;
        }

        let mut node_path = current_path.clone();
        node_path.node_choices.insert(eclass_id, enode.clone());

        match enode {
            // Loops increase the loop level - treat as single unit at level 0
            TileLang::Loop(_) => {
                if loop_level == 0 {
                    results.push(SemiExpression {
                        structure: "(loop)".to_string(),
                        kernel_count: 1,
                        path: node_path,
                    });
                } else {
                    results.push(SemiExpression {
                        structure: "(nested_loop)".to_string(),
                        kernel_count: 0,
                        path: node_path,
                    });
                }
            }

            // Seq nodes: structure matters but don't count as kernels
            TileLang::Seq([left, right]) => {
                let left_semis = enumerate_semi_recursive_with_path(
                    egraph,
                    *left,
                    visited,
                    loop_level,
                    &mut node_path.clone(),
                    Some(enode),
                    0,
                );
                let right_semis = enumerate_semi_recursive_with_path(
                    egraph,
                    *right,
                    visited,
                    loop_level,
                    &mut node_path.clone(),
                    Some(enode),
                    1,
                );

                for left_semi in &left_semis {
                    for right_semi in &right_semis {
                        let mut combined_path = node_path.clone();

                        combined_path
                            .node_choices
                            .extend(left_semi.path.node_choices.clone());
                        combined_path
                            .node_choices
                            .extend(right_semi.path.node_choices.clone());
                        combined_path
                            .traversal_order
                            .extend(left_semi.path.traversal_order.clone());
                        combined_path
                            .traversal_order
                            .extend(right_semi.path.traversal_order.clone());

                        results.push(SemiExpression {
                            structure: format!(
                                "(seq {} {})",
                                left_semi.structure, right_semi.structure
                            ),
                            kernel_count: left_semi.kernel_count + right_semi.kernel_count,
                            path: combined_path,
                        });
                    }
                }
            }
            // Store nodes: direct store operator without tiling
            TileLang::Store(_) => {
                if loop_level == 0 {
                    results.push(SemiExpression {
                        structure: "(store)".to_string(),
                        kernel_count: 1,
                        path: node_path,
                    });
                } else {
                    results.push(SemiExpression {
                        structure: "(nested_op)".to_string(),
                        kernel_count: 0,
                        path: node_path,
                    });
                }
            }

            _ => {
                panic!(
                    "Unexpected node type in semi-expression enumeration: {:?}",
                    enode
                );
            }
        }
    }

    current_path.traversal_order.pop();
    visited.remove(&eclass_id);
    results
}

pub fn enumerate_full_expressions_from_semi(egraph: &EGraph, path: &StructuralPath) -> Vec<String> {
    let mut visited = HashSet::new();
    let extractor = Extractor::new(egraph, AstSize);

    enumerate_full_recursive(
        egraph,
        path.root_eclass,
        &mut visited,
        0,
        None,
        0,
        &extractor,
        path,
    )
}

pub fn enumerate_full_recursive(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    depth: usize,
    parent_node: Option<&TileLang>,
    child_index: usize,
    extractor: &Extractor<AstSize, TileLang, LoopAnalysis>,
    path_constraint: &StructuralPath,
) -> Vec<String> {
    // Handle cycles
    if visited.contains(&eclass_id) {
        let best_expr = extractor.find_best(eclass_id);
        return vec![format!("{}", best_expr.1)];
    }

    if depth > MAX_DEPTH {
        return vec![format!("depth_limit")];
    }

    visited.insert(eclass_id);
    let mut results = Vec::new();
    let eclass = &egraph[eclass_id];

    // KEY CONSTRAINT: If this eclass has a specific node choice in our path,
    // we MUST use that choice. Otherwise, we can explore all nodes
    let nodes_to_explore = if let Some(chosen_node) = path_constraint.node_choices.get(&eclass_id) {
        vec![chosen_node]
    } else {
        eclass.nodes.iter().collect()
    };

    for enode in nodes_to_explore {
        // FILTER: Skip Seq nodes that are left children of parent Seq nodes
        if let TileLang::Seq(_) = enode {
            if let Some(TileLang::Seq(_)) = parent_node {
                if child_index == 0 {
                    // Left child (index 0)
                    continue;
                }
            }
        }

        if let TileLang::TLoop(_) = enode {
            continue;
        }

        if should_skip_seq_for_commutativity(egraph, eclass_id, enode) {
            continue;
        }

        let children = enode.children();
        if children.is_empty() {
            results.push(format!("{}", enode));
        } else {
            let mut child_expressions = Vec::new();
            for (index, &child_id) in children.iter().enumerate() {
                let child_exprs = enumerate_full_recursive(
                    egraph,
                    child_id,
                    visited,
                    depth + 1,
                    Some(enode),
                    index,
                    extractor,
                    path_constraint,
                );
                child_expressions.push(child_exprs);
            }

            let combinations = cartesian_product(&child_expressions);
            for combo in combinations {
                let expr_str = format_enode_with_children(enode, &combo);
                results.push(expr_str);
            }
        }
    }

    visited.remove(&eclass_id);
    results
}

// Alternative version with BigInt support for very large counts
pub fn count_all_expressions_bigint(egraph: &EGraph, eclass_id: Id) -> BigUint {
    let mut memo = HashMap::new();
    let mut visited = HashSet::new();
    count_recursive_with_parent_bigint(egraph, eclass_id, &mut visited, 0, None, 0, &mut memo)
}

pub fn count_recursive_with_parent_bigint(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    depth: usize,
    parent_node: Option<&TileLang>,
    child_index: usize,
    memo: &mut HashMap<Id, BigUint>,
) -> BigUint {
    // Check memoization first
    if !visited.contains(&eclass_id) {
        if let Some(cached_count) = memo.get(&eclass_id) {
            return cached_count.clone();
        }
    }

    // Handle cycles
    if visited.contains(&eclass_id) {
        return BigUint::from(1u32);
    }

    // Depth limit
    if depth > MAX_DEPTH {
        return BigUint::from(1u32);
    }

    visited.insert(eclass_id);
    let mut total_count = BigUint::from(0u32);
    let eclass = &egraph[eclass_id];

    for enode in &eclass.nodes {
        // Apply filters
        // if let TileLang::Store(_) = enode {
        //     total_count += BigUint::from(1u32);
        //     continue;
        // }

        if let TileLang::Seq(_) = enode {
            if let Some(TileLang::Seq(_)) = parent_node {
                if child_index == 0 {
                    continue;
                }
            }
        }

        if let TileLang::TLoop(_) = enode {
            continue;
        }

        // if should_skip_seq_for_commutativity(egraph, eclass_id, enode) {
        //     continue;
        // }

        let children = enode.children();

        if children.is_empty() {
            total_count += BigUint::from(1u32);
        } else {
            let mut node_count = BigUint::from(1u32);

            for (index, &child_id) in children.iter().enumerate() {
                let child_count = count_recursive_with_parent_bigint(
                    egraph,
                    child_id,
                    visited,
                    depth + 1,
                    Some(enode),
                    index,
                    memo,
                );

                node_count *= child_count;
            }

            total_count += node_count;
        }
    }

    visited.remove(&eclass_id);
    memo.insert(eclass_id, total_count.clone());
    total_count
}

pub fn count_expressions_num_kernel(
    egraph: &EGraph,
    eclass_id: Id,
    top_percentage: f32,
) -> BigUint {
    let semi_expressions = enumerate_semi_expressions(egraph, eclass_id);
    let mut sorted_semi = semi_expressions;
    sorted_semi.sort_by_key(|semi| semi.kernel_count);

    let top_count = ((sorted_semi.len() as f32) * top_percentage).max(1.0) as usize;
    let selected_semi = &sorted_semi[..top_count.min(sorted_semi.len())];

    println!(
        "Selected {} semi-expressions out of {} (top {:.1}%)",
        selected_semi.len(),
        sorted_semi.len(),
        top_percentage * 100.0
    );

    let mut total_expressions = BigUint::from(0u32);
    for semi in selected_semi {
        let path_count = count_full_expressions_from_path(egraph, &semi.path);
        println!(
            "Kernel count: {:?}, path count: {:?}",
            semi.kernel_count, path_count
        );
        total_expressions += path_count;
    }

    total_expressions
}

pub fn count_full_expressions_from_path(egraph: &EGraph, path: &StructuralPath) -> BigUint {
    let mut memo = HashMap::new();
    let mut visited = HashSet::new();

    count_full_recursive(
        egraph,
        path.root_eclass,
        &mut visited,
        0,
        None,
        0,
        path,
        &mut memo,
    )
}

pub fn count_full_recursive(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    depth: usize,
    parent_node: Option<&TileLang>,
    child_index: usize,
    path_constraint: &StructuralPath,
    memo: &mut HashMap<Id, BigUint>,
) -> BigUint {
    // Check memorization first
    // if !visited.contains(&eclass_id) {
    //     if let Some(cached_count) = memo.get(&eclass_id) {
    //         return cached_count.clone();
    //     }
    // }

    // Handle cycles
    if visited.contains(&eclass_id) {
        return BigUint::from(1u32);
    }

    // Depth limit
    if depth > MAX_DEPTH {
        return BigUint::from(1u32);
    }

    visited.insert(eclass_id);
    let mut total_count = BigUint::from(0u32);
    let eclass = &egraph[eclass_id];

    let nodes_to_explore = if let Some(chosen_node) = path_constraint.node_choices.get(&eclass_id) {
        vec![chosen_node]
    } else {
        eclass.nodes.iter().collect()
    };

    for enode in nodes_to_explore {
        if egraph[eclass_id].data.is_deleted.contains(&enode) {
            continue;
        }

        // Apply filters
        if let TileLang::Seq(_) = enode {
            if let Some(TileLang::Seq(_)) = parent_node {
                if child_index == 0 {
                    continue;
                }
            }
        }

        if let TileLang::TLoop(_) = enode {
            continue;
        }

        if should_skip_seq_for_commutativity(egraph, eclass_id, enode) {
            continue;
        }

        let children = enode.children();
        if children.is_empty() {
            total_count += BigUint::from(1u32);
        } else {
            let mut node_count = BigUint::from(1u32);

            for (index, &child_id) in children.iter().enumerate() {
                let child_count = count_full_recursive(
                    egraph,
                    child_id,
                    visited,
                    depth + 1,
                    Some(enode),
                    index,
                    path_constraint,
                    memo,
                );
                node_count *= child_count;
            }

            total_count += node_count;
        }
    }

    visited.remove(&eclass_id);
    memo.insert(eclass_id, total_count.clone());
    total_count
}

pub fn format_enode_with_children(enode: &TileLang, children: &[String]) -> String {
    match enode {
        // Loop constructs
        TileLang::Loop(_) => format!(
            "(loop {} {} {} {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string()),
            children.get(3).unwrap_or(&"?".to_string()),
            children.get(4).unwrap_or(&"?".to_string())
        ),

        TileLang::DLoop(_) => format!(
            "(loop {} {} {} {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string()),
            children.get(3).unwrap_or(&"?".to_string()),
            children.get(4).unwrap_or(&"?".to_string())
        ),

        TileLang::TLoop(_) => format!(
            "(tmp_loop {} {} {} {} {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string()),
            children.get(3).unwrap_or(&"?".to_string()),
            children.get(4).unwrap_or(&"?".to_string()),
            children.get(5).unwrap_or(&"?".to_string())
        ),

        // Tensor operations
        TileLang::Input(_) => format!("(input {})", children.get(0).unwrap_or(&"?".to_string())),

        TileLang::Output(_) => format!("(output {})", children.get(0).unwrap_or(&"?".to_string())),

        TileLang::Tensor(_) => format!("(tensor {})", children.get(0).unwrap_or(&"?".to_string())),

        // Indexing
        TileLang::Tile(_) => format!("(tile {})", children.get(0).unwrap_or(&"?".to_string())),

        TileLang::ConstTile(_) => format!(
            "(const_tile {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        TileLang::FullTile => "(fulltile)".to_string(),
        TileLang::Dummy => "(dummy)".to_string(),

        TileLang::Elem(_) => format!("(elem {})", children.get(0).unwrap_or(&"?".to_string())),

        TileLang::Index(_) => {
            if children.is_empty() {
                "(index)".to_string()
            } else {
                format!("(index {})", children.join(" "))
            }
        }

        // Memory operations
        TileLang::Load(_) => format!(
            "(load {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        TileLang::Store(_) => format!(
            "(store {} {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string())
        ),

        TileLang::Seq(_) => format!(
            "(seq {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        // Constants
        TileLang::Const(_) => format!("(const {})", children.get(0).unwrap_or(&"?".to_string())),

        // Arithmetic operations
        TileLang::Add(_) => format!(
            "(+ {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        TileLang::Sub(_) => format!(
            "(- {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        TileLang::Mul(_) => format!(
            "(* {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        TileLang::Div(_) => format!(
            "(/ {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),
        TileLang::Le(_) => format!(
            "(<= {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),
        TileLang::Max(_) => format!(
            "(max {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),
        TileLang::Min(_) => format!(
            "(min {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        TileLang::Exp(_) => format!("(exp {})", children.get(0).unwrap_or(&"?".to_string())),

        TileLang::Sqr(_) => format!("(sqr {})", children.get(0).unwrap_or(&"?".to_string())),

        TileLang::Sqrt(_) => format!("(sqrt {})", children.get(0).unwrap_or(&"?".to_string())),

        TileLang::Sigmoid(_) => {
            format!("(sigmoid {})", children.get(0).unwrap_or(&"?".to_string()))
        }
        TileLang::Erf(_) => format!("(erf {})", children.get(0).unwrap_or(&"?".to_string())),
        TileLang::Abs(_) => format!("(abs {})", children.get(0).unwrap_or(&"?".to_string())),

        TileLang::Matmul(_) => format!(
            "(@ {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        TileLang::ReduceSum(_) => format!(
            "(rsum {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),
        TileLang::ReduceMin(_) => format!(
            "(rmin {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),
        TileLang::ReduceMax(_) => format!(
            "(rmax {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        // Tensor manipulation
        TileLang::Concat(_) => format!(
            "(concat {} {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string())
        ),

        TileLang::Broadcast(_) => format!(
            "(bcast {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        TileLang::Transpose(_) => {
            format!("(transpose {})", children.get(0).unwrap_or(&"?".to_string()))
        },

        TileLang::Permute3(_) => format!(
            "(permute3 {} {} {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string()),
            children.get(3).unwrap_or(&"?".to_string())
        ),
        TileLang::Permute4(_) => format!(
            "(permute4 {} {} {} {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string()),
            children.get(3).unwrap_or(&"?".to_string()),
            children.get(4).unwrap_or(&"?".to_string())
        ),

        TileLang::Squeeze(_) => format!(
            "(squeeze {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        TileLang::Unsqueeze(_) => format!(
            "(unsqueeze {} {})",
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())
        ),

        // Leaf nodes
        TileLang::Num(n) => n.to_string(),
        TileLang::Var(s) => s.to_string(),
        _ => "".to_string(),
    }
}
// Helper function to compute cartesian product
pub fn cartesian_product<T: Clone>(lists: &[Vec<T>]) -> Vec<Vec<T>> {
    if lists.is_empty() {
        return vec![vec![]];
    }

    if lists.len() == 1 {
        return lists[0].iter().map(|item| vec![item.clone()]).collect();
    }

    let mut result = Vec::new();
    let first = &lists[0];
    let rest_product = cartesian_product(&lists[1..]);

    for item in first {
        for rest in &rest_product {
            let mut combination = vec![item.clone()];
            combination.extend(rest.clone());
            result.push(combination);
        }
    }

    result
}

/// Helper function to check if we should skip this Seq node due to commutativity
pub fn should_skip_seq_for_commutativity(
    egraph: &EGraph,
    eclass_id: Id,
    current_seq: &TileLang,
) -> bool {
    if let TileLang::Seq([left, right]) = current_seq {
        // Case1: Skip (seq a b) vs (seq b a)
        for other_enode in &egraph[eclass_id].nodes {
            if let TileLang::Seq([other_left, other_right]) = other_enode {
                // Skip if we find the commuted version and our current seq has "larger" ordering
                if *left == *other_right && *right == *other_left {
                    // Use canonical ordering: only process the seq where left_id <= right_id
                    if *left > *right {
                        return true; // Skip this one, process the other
                    }
                }
            }
        }

        // Case2: Skip (seq a (seq b others)) vs (seq b (seq a others))
        let right_eclass = &egraph[*right];
        for right_enode in &right_eclass.nodes {
            if let TileLang::Seq([nested_left, nested_right]) = right_enode {
                // Look for (seq nested_left (seq left nested_right))
                for other_enode in &egraph[eclass_id].nodes {
                    if let TileLang::Seq([other_left, other_right]) = other_enode {
                        if *other_left == *nested_left {
                            // Check if other_right represents (seq left nested_right)
                            let other_right_eclass = &egraph[*other_right];
                            for other_right_enode in &other_right_eclass.nodes {
                                if let TileLang::Seq([other_nested_left, other_nested_right]) =
                                    other_right_enode
                                {
                                    if *other_nested_left == *left
                                        && *other_nested_right == *nested_right
                                    {
                                        // Found pattern: (seq left (seq nested_left nested_right)) vs (seq nested_left (seq left nested_right))
                                        if *left > *nested_left {
                                            return true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

// Function to return expressions as strings for easier reading
pub fn enumerate_expressions_all_as_strings(egraph: &EGraph, eclass_id: Id) -> Vec<String> {
    let expressions = enumerate_expressions_all(egraph, eclass_id); // strawman
    expressions.iter().map(|expr| expr.to_string()).collect()
}

// List all possible expressions from the given eclass
pub fn list_expressions_all(runner: &egg::Runner<TileLang, LoopAnalysis>) -> Vec<String> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0]; // Assuming single root expression
    enumerate_expressions_all_as_strings(egraph, root_id)
}
// Count all possible expressions from the given eclass
pub fn count_expressions_all_for_root(runner: &egg::Runner<TileLang, LoopAnalysis>) -> BigUint {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0]; // Assuming single root expression

    count_all_expressions_bigint(egraph, root_id) // strawman
}

// List expressions with minimum number of kernel from the given eclass
pub fn list_expressions_num_kernel(
    runner: &egg::Runner<TileLang, LoopAnalysis>,
    top_percentage: f32,
) -> Vec<String> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0]; // Assuming single root expression
    enumerate_expressions_num_kernel(egraph, root_id, top_percentage)
}

// Count expressions with minimum number of kernel from the given eclass
pub fn count_expressions_num_kernel_for_root(
    runner: &egg::Runner<TileLang, LoopAnalysis>,
    top_percentage: f32,
) -> BigUint {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0]; // Assuming single root expression

    count_expressions_num_kernel(egraph, root_id, top_percentage) // strawman
}
