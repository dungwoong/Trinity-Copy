use crate::cost::*;
use crate::extract::*;
use crate::language::*;
use crate::utils::*;
use egg::*;
use json::{object, JsonValue};
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs;
use std::io;

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;
const MAX_DEPTH: usize = 100;

#[derive(Debug, Clone)]
pub struct KernelCost {
    pub cost: usize,
    pub num_kernel: usize,
}

impl KernelCost {
    pub fn new(cost: usize, num_kernel: usize) -> Self {
        KernelCost { cost, num_kernel }
    }

    pub fn zero() -> Self {
        KernelCost {
            cost: 0,
            num_kernel: 0,
        }
    }

    pub fn add(&self, other: &KernelCost) -> KernelCost {
        KernelCost {
            cost: self.cost + other.cost,
            num_kernel: self.num_kernel + other.num_kernel,
        }
    }

    pub fn is_within_limits(&self, max_cost: usize, max_num_kernel: usize) -> bool {
        self.cost <= max_cost && self.num_kernel <= max_num_kernel
    }

    fn to_json(&self) -> JsonValue {
        object! {
            cost: self.cost,
            num_kernel: self.num_kernel
        }
    }

    // Parse from JSON object
    fn from_json(json: &JsonValue) -> Result<Self, String> {
        let cost = json["cost"]
            .as_usize()
            .ok_or("Missing or invalid 'cost' field")?;
        let num_kernel = json["num_kernel"]
            .as_usize()
            .ok_or("Missing or invalid 'num_kernel' field")?;

        Ok(KernelCost::new(cost, num_kernel))
    }
}

#[derive(Debug, Clone)]
pub struct NodeChoice {
    pub eclass_id: Id,
    pub enode_index: usize,
}

impl NodeChoice {
    // Convert to JSON object
    fn to_json(&self) -> JsonValue {
        object! {
            eclass_id: usize::from(self.eclass_id),
            enode_index: self.enode_index
        }
    }

    // Parse from JSON object
    fn from_json(json: &JsonValue) -> Result<Self, String> {
        let eclass_id = json["eclass_id"]
            .as_usize()
            .ok_or("Missing or invalid 'eclass_id' field")?;
        let enode_index = json["enode_index"]
            .as_usize()
            .ok_or("Missing or invalid 'enode_index' field")?;

        Ok(NodeChoice {
            eclass_id: egg::Id::from(eclass_id),
            enode_index,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SemiExpressionResult {
    pub semi_expression: String,
    pub cost: KernelCost,
    pub choices: Vec<NodeChoice>,
}

impl SemiExpressionResult {
    // Convert to JSON object
    fn to_json(&self) -> JsonValue {
        let mut choices_json = JsonValue::new_array();
        for choice in &self.choices {
            choices_json.push(choice.to_json()).unwrap();
        }

        object! {
            semi_expression: self.semi_expression.clone(),
            cost: self.cost.to_json(),
            choices: choices_json
        }
    }

    // Parse from JSON object
    fn from_json(json: &JsonValue) -> Result<Self, String> {
        let semi_expression = json["semi_expression"]
            .as_str()
            .ok_or("Missing or invalid 'semi_expression' field")?
            .to_string();

        let cost = KernelCost::from_json(&json["cost"])?;

        let mut choices = Vec::new();
        if json["choices"].is_array() {
            for i in 0..json["choices"].len() {
                // Use array indexing instead of get()
                let choice_json = &json["choices"][i];
                choices.push(NodeChoice::from_json(choice_json)?);
            }
        }

        Ok(SemiExpressionResult {
            semi_expression,
            cost,
            choices,
        })
    }
}

// Enumerate expressions with cost constraint
pub fn enumerate_expressions_with_target_cost_v2(
    egraph: &EGraph,
    eclass_id: Id,
    max_cost: usize,
    max_num_kernel: usize,
) -> Vec<SemiExpressionResult> {
    let mut visited = HashMap::new();
    // let mut visited = HashSet::new();
    // let extractor = Extractor::new(egraph, AstSize);
    let extractor = create_extractor(egraph);
    let cost_model = TileCostModel::new();

    let expr_pairs = enumerate_recursive_with_cost_v2(
        egraph,
        eclass_id,
        &mut visited,
        0,
        None,
        0,
        &extractor,
        &cost_model,
        max_cost,
        max_num_kernel,
        0,     // loop level
        false, // should_sequential
    );

    expr_pairs
}

pub fn enumerate_recursive_with_cost_v2(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashMap<Id, usize>,
    // visited: &mut HashSet<Id>,
    depth: usize,
    parent_node: Option<&TileLang>,
    child_index: usize,
    extractor: &Extractor<TileCost, TileLang, LoopAnalysis>,
    cost_model: &TileCostModel,
    max_cost: usize,
    max_num_kernel: usize,
    loop_level: usize,
    should_sequential: bool,
) -> Vec<SemiExpressionResult> {
    // Original implementation - filter on first visit
    // if visited.contains(&eclass_id) {
    //     let (best_cost, best_expr) = extractor.find_best(eclass_id);
    //     if best_cost <= max_cost {
    //         return vec![SemiExpressionResult {
    //             semi_expression: format!("{}", best_expr),
    //             cost: KernelCost::new(best_cost, 0),
    //             choices: vec![],
    //         }]
    //     } else {
    //         return vec![];
    //     }
    // }

    // New implementation - allow one cycle before filtering
    let visit_count = visited.get(&eclass_id).copied().unwrap_or(0);
    if visit_count >= 1 {
        let (best_cost, best_expr) = extractor.find_best(eclass_id);
        if best_cost <= max_cost {
            return vec![SemiExpressionResult {
                semi_expression: format!("{}", best_expr),
                cost: KernelCost::new(best_cost, 0),
                choices: vec![],
            }];
        } else {
            return vec![];
        }
    }

    if depth > MAX_DEPTH {
        return vec![SemiExpressionResult {
            semi_expression: format!("depth_limit"),
            cost: KernelCost::zero(),
            choices: vec![],
        }];
    }

    // Increment visit count
    // visited.insert(eclass_id);
    *visited.entry(eclass_id).or_insert(0) += 1;
    let mut results = Vec::new();
    let eclass = &egraph[eclass_id];

    for (enode_index, enode) in eclass.nodes.iter().enumerate() {
        if let TileLang::Store(_) = enode {
            // Calculate kernel count for Store node
            let store_kernel_count = if loop_level == 0 { 1 } else { 0 };
            let store_cost_struct = KernelCost::new(0, store_kernel_count);

            // Check if within limits
            if !store_cost_struct.is_within_limits(max_cost, max_num_kernel) {
                continue;
            }

            let choice = NodeChoice {
                eclass_id,
                enode_index,
            };
            results.push(SemiExpressionResult {
                semi_expression: format!("(store)"),
                cost: store_cost_struct,
                // cost: KernelCost::zero(),
                choices: vec![choice],
            });
            continue;
        }

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

        // Calculate cost for this node
        let node_cost = if loop_level >= 1 && should_sequential {
            cost_model.get_self_cost_should_sequential(egraph, enode, true)
        } else {
            cost_model.get_self_cost_should_sequential(egraph, enode, false)
        };

        // Calculate kernel count for this node
        let node_kernel_count = match enode {
            TileLang::Loop(_) | TileLang::Store(_) => {
                if loop_level == 0 {
                    1
                } else {
                    0
                }
            }
            _ => 0,
        };

        let node_cost_struct = KernelCost::new(node_cost, node_kernel_count);
        if !node_cost_struct.is_within_limits(max_cost, max_num_kernel) {
            continue;
        }

        // Record this node choice
        let current_choice = NodeChoice {
            eclass_id,
            enode_index,
        };

        let (new_loop_level, new_should_sequential) = match enode {
            TileLang::Loop(_) => (loop_level + 1, should_sequential),
            TileLang::Seq(_) => (loop_level, true),
            _ => (loop_level, should_sequential),
        };

        let children = enode.children();

        if children.is_empty() {
            results.push(SemiExpressionResult {
                semi_expression: format!("{}", enode),
                cost: node_cost_struct,
                choices: vec![current_choice],
            });
        } else {
            let remaining_cost = max_cost - node_cost;
            let remaining_kernel = max_num_kernel - node_kernel_count;
            let mut child_semi_results = Vec::new();
            let mut all_children_valid = true;

            for (index, &child_id) in children.iter().enumerate() {
                let child_results = enumerate_recursive_with_cost_v2(
                    egraph,
                    child_id,
                    visited,
                    depth + 1,
                    Some(enode),
                    index,
                    extractor,
                    cost_model,
                    remaining_cost,
                    remaining_kernel,
                    new_loop_level,
                    new_should_sequential,
                );

                if child_results.is_empty() {
                    all_children_valid = false;
                    break;
                }

                child_semi_results.push(child_results);
            }

            if all_children_valid {
                let combinations = cartesian_product(&child_semi_results);
                for combo in combinations {
                    let mut total_cost = node_cost_struct.clone();
                    let mut all_choices = vec![current_choice.clone()];

                    for semi_result in &combo {
                        total_cost = total_cost.add(&semi_result.cost);
                        all_choices.extend(semi_result.choices.clone());
                    }

                    if total_cost.is_within_limits(max_cost, max_num_kernel) {
                        let child_exprs: Vec<String> = combo
                            .iter()
                            .map(|result| result.semi_expression.clone())
                            .collect();
                        let expr_str = format_enode_with_children(enode, &child_exprs);

                        results.push(SemiExpressionResult {
                            semi_expression: expr_str,
                            cost: total_cost,
                            choices: all_choices,
                        });
                    }
                }
            }
        }
    }
    visited.remove(&eclass_id);
    results
}

pub fn reconstruct_full_expression_naive(
    egraph: &EGraph,
    root_id: Id,
    choices: &[NodeChoice],
) -> Vec<String> {
    let mut visited = HashSet::new();
    let extractor = create_extractor(egraph);
    let mut loop_variables = HashSet::new();

    let choice_map: HashMap<Id, usize> = choices
        .iter()
        .map(|choice| (choice.eclass_id, choice.enode_index))
        .collect();

    reconstruct_full_recursive_naive(
        egraph,
        root_id,
        &mut visited,
        0,
        None,
        0,
        &extractor,
        &choice_map,
        &mut loop_variables,
    )
}

pub fn reconstruct_full_recursive_naive(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    depth: usize,
    parent_node: Option<&TileLang>,
    child_index: usize,
    extractor: &Extractor<TileCost, TileLang, LoopAnalysis>,
    choice_constraints: &HashMap<Id, usize>,
    loop_variables: &mut HashSet<String>,
) -> Vec<String> {
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

    let nodes_to_explore = if let Some(&chosen_enode_index) = choice_constraints.get(&eclass_id) {
        vec![&eclass.nodes[chosen_enode_index]]
    } else {
        eclass.nodes.iter().collect()
    };

    for enode in nodes_to_explore {
        match enode {
            TileLang::Tile(var_id) => {
                let Some(var_name) = get_var_string(egraph, *var_id) else {
                    panic!("failed to get var string {}", var_id);
                };
                if !loop_variables.contains(&var_name) {
                    continue;
                }
            }
            TileLang::Elem(var_id) => {
                let Some(var_name) = get_var_string(egraph, *var_id) else {
                    panic!("failed to get var string {}", var_id);
                };
                if !loop_variables.contains(&var_name) {
                    continue;
                }
            }
            _ => {}
        }

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

        let loop_var_to_add = if let TileLang::Loop([_start, _end, _tile, loop_var, _body]) = enode
        {
            get_var_string(egraph, *loop_var)
        } else {
            None
        };

        let children = enode.children();
        if children.is_empty() {
            results.push(format!("{}", enode));
        } else {
            let mut child_expressions = Vec::new();
            let mut all_children_valid = true;
            for (index, &child_id) in children.iter().enumerate() {
                if let Some(ref var) = loop_var_to_add {
                    loop_variables.insert(var.clone());
                }

                let child_exprs = reconstruct_full_recursive_naive(
                    egraph,
                    child_id,
                    visited,
                    depth + 1,
                    Some(enode),
                    index,
                    extractor,
                    choice_constraints,
                    loop_variables,
                );

                // Remove loop variable after processing body
                if let Some(ref var) = loop_var_to_add {
                    loop_variables.remove(var);
                }

                if child_exprs.is_empty() {
                    all_children_valid = false;
                    break;
                }

                child_expressions.push(child_exprs);
            }

            if all_children_valid {
                let combinations = cartesian_product(&child_expressions);
                for combo in combinations {
                    let expr_str = format_enode_with_children(enode, &combo);
                    results.push(expr_str);
                }
            }
        }
        break;
    }

    visited.remove(&eclass_id);
    return results;

    // if let Some(&chosen_enode_index) = choice_constraints.get(&eclass_id) {
    //     let enodes_to_explore = vec![(chosen_enode_index, &eclass.nodes[chosen_enode_index])];
    //     for (enode_index, enode) in enodes_to_explore {
    //         match enode {
    //             TileLang::Tile(var_id) => {
    //                 let Some(var_name) = get_var_string(egraph, *var_id) else {
    //                     panic!("failed to get var string {}", var_id);
    //                     continue;
    //                 };
    //                 if !loop_variables.contains(&var_name) {
    //                     continue;
    //                 }
    //             }
    //             TileLang::Elem(var_id) => {
    //                 let Some(var_name) = get_var_string(egraph, *var_id) else {
    //                     panic!("failed to get var string {}", var_id);
    //                     continue;
    //                 };
    //                 if !loop_variables.contains(&var_name) {
    //                     continue;
    //                 }
    //             }
    //             _ => {}
    //         }

    //         if let TileLang::Seq(_) = enode {
    //             if let Some(TileLang::Seq(_)) = parent_node {
    //                 if child_index == 0 { // Left child (index 0)
    //                     continue;
    //                 }
    //             }
    //         }

    //         if let TileLang::TLoop(_) = enode {
    //             continue;
    //         }

    //         if should_skip_seq_for_commutativity(egraph, eclass_id, enode) {
    //             continue;
    //         }

    //         let loop_var_to_add = if let TileLang::Loop([_start, _end, _tile, loop_var, _body]) = enode {
    //             get_var_string(egraph, *loop_var)
    //         } else {
    //             None
    //         };

    //         let children = enode.children();
    //         if children.is_empty() {
    //             results.push(format!("{}", enode));
    //         } else {
    //             let mut child_expressions = Vec::new();
    //             let mut all_children_valid = true;
    //             for (index, &child_id) in children.iter().enumerate() {

    //                 if let Some(ref var) = loop_var_to_add {
    //                     loop_variables.insert(var.clone());
    //                 }

    //                 let child_exprs = reconstruct_full_recursive_naive(
    //                     egraph,
    //                     child_id,
    //                     visited,
    //                     depth+1,
    //                     Some(enode),
    //                     index,
    //                     extractor,
    //                     choice_constraints,
    //                     loop_variables,
    //                 );

    //                 // Remove loop variable after processing body
    //                 if let Some(ref var) = loop_var_to_add {
    //                     loop_variables.remove(var);
    //                 }

    //                 if child_exprs.is_empty() {
    //                     all_children_valid = false;
    //                     break;
    //                 }

    //                 child_expressions.push(child_exprs);
    //             }

    //             if all_children_valid {
    //                 let combinations = cartesian_product(&child_expressions);
    //                 for combo in combinations {
    //                     let expr_str = format_enode_with_children(enode, &combo);
    //                     results.push(expr_str);
    //                 }
    //             }
    //         }

    //     }
    //     visited.remove(&eclass_id);
    //     return results;
    // } else {
    //     // let best_expr = extractor.find_best(eclass_id);
    //     let ast_extractor = Extractor::new(egraph, AstSize);
    //     let best_expr = ast_extractor.find_best(eclass_id);
    //     visited.remove(&eclass_id);
    //     return vec![format!("{}", best_expr.1)];
    // }
}

pub fn reconstruct_full_expression_all(
    egraph: &EGraph,
    root_id: Id,
    choices: &[NodeChoice],
) -> Vec<String> {
    let mut visited = HashSet::new();
    let extractor = create_extractor(egraph);
    let mut loop_variables = HashSet::new();

    let choice_map: HashMap<Id, usize> = choices
        .iter()
        .map(|choice| (choice.eclass_id, choice.enode_index))
        .collect();

    reconstruct_full_recursive_all(
        egraph,
        root_id,
        &mut visited,
        0,
        None,
        0,
        &extractor,
        &choice_map,
        &mut loop_variables,
    )
}

pub fn reconstruct_full_recursive_all(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    depth: usize,
    parent_node: Option<&TileLang>,
    child_index: usize,
    extractor: &Extractor<TileCost, TileLang, LoopAnalysis>,
    choice_constraints: &HashMap<Id, usize>,
    loop_variables: &mut HashSet<String>,
) -> Vec<String> {
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

    let enodes_to_explore: Vec<(usize, &TileLang)> =
        if let Some(&chosen_enode_index) = choice_constraints.get(&eclass_id) {
            vec![(chosen_enode_index, &eclass.nodes[chosen_enode_index])]
        } else {
            eclass.nodes.iter().enumerate().collect()
        };

    for (_enode_index, enode) in enodes_to_explore {
        match enode {
            TileLang::Tile(var_id) => {
                let Some(var_name) = get_var_string(egraph, *var_id) else {
                    panic!("failed to get var string {}", var_id);
                };
                if !loop_variables.contains(&var_name) {
                    continue;
                }
            }
            TileLang::Elem(var_id) => {
                let Some(var_name) = get_var_string(egraph, *var_id) else {
                    panic!("failed to get var string {}", var_id);
                };
                if !loop_variables.contains(&var_name) {
                    continue;
                }
            }
            _ => {}
        }

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

        let loop_var_to_add = if let TileLang::Loop([_start, _end, _tile, loop_var, _body]) = enode
        {
            get_var_string(egraph, *loop_var)
        } else {
            None
        };

        let children = enode.children();
        if children.is_empty() {
            results.push(format!("{}", enode));
        } else {
            let mut child_expressions = Vec::new();
            let mut all_children_valid = true;
            for (index, &child_id) in children.iter().enumerate() {
                if let Some(ref var) = loop_var_to_add {
                    loop_variables.insert(var.clone());
                }

                let child_exprs = reconstruct_full_recursive_all(
                    egraph,
                    child_id,
                    visited,
                    depth + 1,
                    Some(enode),
                    index,
                    extractor,
                    choice_constraints,
                    loop_variables,
                );

                // Remove loop variable after processing body
                if let Some(ref var) = loop_var_to_add {
                    loop_variables.remove(var);
                }

                if child_exprs.is_empty() {
                    all_children_valid = false;
                    break;
                }

                child_expressions.push(child_exprs);
            }

            if all_children_valid {
                let combinations = cartesian_product(&child_expressions);
                for combo in combinations {
                    let expr_str = format_enode_with_children(enode, &combo);
                    results.push(expr_str);
                }
            }
        }
    }

    visited.remove(&eclass_id);
    results
}

pub fn create_egraph_from_semi_expression(
    original_egraph: &EGraph,
    semi_result: &SemiExpressionResult,
    root_eclass_id: Id,
) -> (EGraph, Id) {
    let mut new_egraph = EGraph::new(LoopAnalysis);

    // Create a map of eclass_id -> chosen enode_index from the choices
    let mut choices_map: HashMap<Id, usize> = HashMap::new();
    for choice in &semi_result.choices {
        choices_map.insert(choice.eclass_id, choice.enode_index);
    }

    // Use all eclasses from the original egraph to avoid missing dependencies
    // let all_eclasses: Vec<Id> = original_egraph.classes().map(|c| c.id).collect();
    let all_eclasses = find_reachable_eclasses(original_egraph, root_eclass_id, &choices_map);

    // Create mapping using unique dummy nodes for each eclass
    let mut old_to_new_id: HashMap<Id, Id> = HashMap::new();
    for (i, &eclass_id) in all_eclasses.iter().enumerate() {
        // Use unique variable names to ensure different eclasses
        let dummy_var_name = format!("dummydata{}", i);
        let new_id = new_egraph.add_uncanonical(TileLang::Var(dummy_var_name.parse().unwrap()));
        old_to_new_id.insert(eclass_id, new_id);
    }

    // Now add the actual nodes to each eclass
    for &eclass_id in &all_eclasses {
        let class = &original_egraph[eclass_id];
        let new_class_id = old_to_new_id[&eclass_id];

        if let Some(&chosen_index) = choices_map.get(&eclass_id) {
            // Add only the chosen node
            if chosen_index < class.nodes.len() {
                let chosen_node = &class.nodes[chosen_index];
                let updated_node = update_node_children(chosen_node, &old_to_new_id);
                let node_id = new_egraph.add(updated_node);
                new_egraph.union(new_class_id, node_id);
            } else {
                panic!("No chosen node!");
            }
        } else {
            // Add all nodes from the original eclass
            for node in &class.nodes {
                let updated_node = update_node_children(node, &old_to_new_id);
                let node_id = new_egraph.add(updated_node);
                new_egraph.union(new_class_id, node_id);
            }
        }
    }

    new_egraph.rebuild();

    // Copy tensor shape information from original egraph
    for &eclass_id in &all_eclasses {
        let original_class = &original_egraph[eclass_id];
        let new_class_id = old_to_new_id[&eclass_id];

        // If the original eclass has shape information, copy it to the new egraph
        if let Some(tensor_shape) = &original_class.data.tensor_shape {
            let new_class = &mut new_egraph[new_class_id];
            new_class.data.tensor_shape = Some(tensor_shape.clone());
        }
    }

    new_egraph.rebuild();

    let new_root_id = old_to_new_id[&root_eclass_id];
    (new_egraph, new_root_id)
}

fn find_reachable_eclasses(
    egraph: &EGraph,
    root_id: Id,
    choices_map: &HashMap<Id, usize>,
) -> HashSet<Id> {
    let mut reachable = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back(root_id);
    reachable.insert(root_id);

    while let Some(current_id) = queue.pop_front() {
        let class = &egraph[current_id];

        if let Some(&chosen_index) = choices_map.get(&current_id) {
            let chosen_node = &class.nodes[chosen_index];
            for &child_id in chosen_node.children() {
                if !reachable.contains(&child_id) {
                    reachable.insert(child_id);
                    queue.push_back(child_id);
                }
            }
        } else {
            for node in &class.nodes {
                for &child_id in node.children() {
                    if !reachable.contains(&child_id) {
                        reachable.insert(child_id);
                        queue.push_back(child_id);
                    }
                }
            }
        }
    }

    reachable
}

fn update_node_children(node: &TileLang, old_to_new_id: &HashMap<Id, Id>) -> TileLang {
    match node {
        TileLang::Loop(children) => TileLang::Loop([
            old_to_new_id[&children[0]],
            old_to_new_id[&children[1]],
            old_to_new_id[&children[2]],
            old_to_new_id[&children[3]],
            old_to_new_id[&children[4]],
        ]),
        TileLang::DLoop(children) => TileLang::Loop([
            old_to_new_id[&children[0]],
            old_to_new_id[&children[1]],
            old_to_new_id[&children[2]],
            old_to_new_id[&children[3]],
            old_to_new_id[&children[4]],
        ]),
        TileLang::Input(child) => TileLang::Input(old_to_new_id[child]),
        TileLang::Output(child) => TileLang::Output(old_to_new_id[child]),
        TileLang::Tensor(child) => TileLang::Tensor(old_to_new_id[child]),
        TileLang::Tile(child) => TileLang::Tile(old_to_new_id[child]),
        TileLang::ConstTile(children) => {
            TileLang::ConstTile([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Elem(child) => TileLang::Elem(old_to_new_id[child]),
        TileLang::Index(children) => {
            let updated_children: Vec<Id> = children
                .iter()
                .map(|&child| old_to_new_id[&child])
                .collect();
            TileLang::Index(updated_children.into_boxed_slice())
        }
        TileLang::Load(children) => {
            TileLang::Load([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Store(children) => TileLang::Store([
            old_to_new_id[&children[0]],
            old_to_new_id[&children[1]],
            old_to_new_id[&children[2]],
        ]),
        TileLang::Seq(children) => {
            TileLang::Seq([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Const(child) => TileLang::Const(old_to_new_id[child]),
        TileLang::Add(children) => {
            TileLang::Add([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Sub(children) => {
            TileLang::Sub([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Mul(children) => {
            TileLang::Mul([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Div(children) => {
            TileLang::Div([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Le(children) => {
            TileLang::Le([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Max(children) => {
            TileLang::Max([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Min(children) => {
            TileLang::Min([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Exp(child) => TileLang::Exp(old_to_new_id[child]),
        TileLang::Sqr(child) => TileLang::Sqr(old_to_new_id[child]),
        TileLang::Sqrt(child) => TileLang::Sqrt(old_to_new_id[child]),
        TileLang::Sigmoid(child) => TileLang::Sigmoid(old_to_new_id[child]),
        TileLang::Erf(child) => TileLang::Erf(old_to_new_id[child]),
        TileLang::Abs(child) => TileLang::Abs(old_to_new_id[child]),
        TileLang::Matmul(children) => {
            TileLang::Matmul([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::ReduceSum(children) => {
            TileLang::ReduceSum([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::ReduceMin(children) => {
            TileLang::ReduceMin([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::ReduceMax(children) => {
            TileLang::ReduceMax([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Concat(children) => TileLang::Concat([
            old_to_new_id[&children[0]],
            old_to_new_id[&children[1]],
            old_to_new_id[&children[2]],
        ]),
        TileLang::Broadcast(children) => {
            TileLang::Broadcast([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Transpose(child) => {
            TileLang::Transpose(
                old_to_new_id[child]
            )      
        },
        TileLang::Permute3(children) => TileLang::Permute3([
            old_to_new_id[&children[0]],
            old_to_new_id[&children[1]],
            old_to_new_id[&children[2]],
            old_to_new_id[&children[3]],
        ]),
        TileLang::Permute4(children) => TileLang::Permute4([
            old_to_new_id[&children[0]],
            old_to_new_id[&children[1]],
            old_to_new_id[&children[2]],
            old_to_new_id[&children[3]],
            old_to_new_id[&children[4]],
        ]),
        TileLang::Squeeze(children) => {
            TileLang::Squeeze([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        TileLang::Unsqueeze(children) => {
            TileLang::Unsqueeze([old_to_new_id[&children[0]], old_to_new_id[&children[1]]])
        }
        // Leaf nodes (no children)
        TileLang::FullTile => TileLang::FullTile,
        TileLang::Dummy => TileLang::Dummy,
        TileLang::Num(n) => TileLang::Num(*n),
        TileLang::Var(v) => TileLang::Var(*v),
        _ => TileLang::Dummy,
    }
}

// Enumerate expressions with cost constraint
pub fn enumerate_expressions_with_target_cost(
    egraph: &EGraph,
    eclass_id: Id,
    max_cost: usize,
    max_num_kernel: usize,
) -> Vec<String> {
    let mut visited = HashSet::new();
    // let extractor = Extractor::new(egraph, AstSize);
    let extractor = create_extractor(egraph);
    let cost_model = TileCostModel::new();

    let expr_pairs = enumerate_recursive_with_cost(
        egraph,
        eclass_id,
        &mut visited,
        0,
        None,
        0,
        &extractor,
        &cost_model,
        max_cost,
        max_num_kernel,
        0,     // loop level
        false, // should_sequential
    );

    expr_pairs.into_iter().map(|(expr, _)| expr).collect()
}

pub fn enumerate_recursive_with_cost(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    depth: usize,
    parent_node: Option<&TileLang>,
    child_index: usize,
    extractor: &Extractor<TileCost, TileLang, LoopAnalysis>,
    cost_model: &TileCostModel,
    max_cost: usize,
    max_num_kernel: usize,
    loop_level: usize,
    should_sequential: bool,
) -> Vec<(String, KernelCost)> {
    // Handle cycles - allow only once
    if visited.contains(&eclass_id) {
        // Only return best expression if current cost is within limit
        let (best_cost, best_expr) = extractor.find_best(eclass_id);
        if best_cost <= max_cost {
            return vec![(format!("{}", best_expr), KernelCost::new(best_cost, 0))];
        } else {
            return vec![];
        }
    }

    // Depth limit
    if depth > MAX_DEPTH {
        return vec![(format!("depth_limit"), KernelCost::zero())];
    }

    visited.insert(eclass_id);
    let mut results = Vec::new();
    let eclass = &egraph[eclass_id];

    for enode in &eclass.nodes {
        // Stop traversing for efficiency
        if let TileLang::Store(_) = enode {
            results.push((format!("(store)"), KernelCost::zero()));
            continue;
        }

        // Apply existing filters
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

        // Calculate cost for this node
        let node_cost = if loop_level >= 1 && should_sequential {
            cost_model.get_self_cost_should_sequential(egraph, enode, true)
        } else {
            cost_model.get_self_cost_should_sequential(egraph, enode, false)
        };

        // Calculate kernel count for this node
        let node_kernel_count = if let TileLang::Loop(_) = enode {
            if loop_level == 0 {
                1
            } else {
                0
            }
        } else {
            0
        };

        let node_cost_struct = KernelCost::new(node_cost, node_kernel_count);
        if !node_cost_struct.is_within_limits(max_cost, max_num_kernel) {
            continue;
        }

        // Determine new loop level and should sequential
        let (new_loop_level, new_should_sequential) = match enode {
            TileLang::Loop(_) => (loop_level + 1, should_sequential),
            TileLang::Seq(_) => (loop_level, true),
            _ => (loop_level, should_sequential),
        };

        let children = enode.children();

        if children.is_empty() {
            // Leaf node - include if cost is within limit
            results.push((format!("{}", enode), node_cost_struct));
        } else {
            // Get expressions for each child with updated cost
            let remaining_cost = max_cost - node_cost;
            let remaining_kernel = max_num_kernel - node_kernel_count;
            let mut child_expr_cost_pairs = Vec::new();
            let mut all_children_valid = true;

            for (index, &child_id) in children.iter().enumerate() {
                let child_pairs = enumerate_recursive_with_cost(
                    egraph,
                    child_id,
                    visited,
                    depth + 1,
                    Some(enode),
                    index,
                    extractor,
                    cost_model,
                    remaining_cost,
                    remaining_kernel,
                    new_loop_level,
                    new_should_sequential,
                );

                if child_pairs.is_empty() {
                    all_children_valid = false;
                    break;
                }

                child_expr_cost_pairs.push(child_pairs);
            }

            // Only generate combinations if all children have valid expressions
            if all_children_valid {
                let combinations = cartesian_product(&child_expr_cost_pairs);
                for combo in combinations {
                    let mut total_cost = node_cost_struct.clone();
                    for (_, child_cost) in &combo {
                        total_cost = total_cost.add(child_cost);
                    }

                    if total_cost.is_within_limits(max_cost, max_num_kernel) {
                        let child_exprs: Vec<String> =
                            combo.iter().map(|(expr, _)| expr.clone()).collect();
                        let expr_str = format_enode_with_children(enode, &child_exprs);
                        results.push((expr_str, total_cost));
                    }
                }
            }
        }
    }

    visited.remove(&eclass_id);
    results
}

// List expressions with minimum number of kernel from the given eclass
pub fn list_expressions_with_target_cost(
    runner: &egg::Runner<TileLang, LoopAnalysis>,
) -> Vec<String> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0]; // Assuming single root expression

    let extractor = create_extractor(egraph);
    let (min_cost, _) = extractor.find_best(root_id);

    let max_cost = 2;
    let max_num_kernel = 2;

    println!("Minimum cost: {:?}", min_cost.clone());
    enumerate_expressions_with_target_cost(egraph, root_id, max_cost, max_num_kernel)
}

pub fn list_expressions_with_target_cost_v2(
    runner: &egg::Runner<TileLang, LoopAnalysis>,
) -> Vec<String> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0];

    let max_cost = 3;
    let max_num_kernel = 1;

    let semi_results =
        enumerate_expressions_with_target_cost_v2(egraph, root_id, max_cost, max_num_kernel);

    println!("There are {} semi-expressions", &semi_results.len());

    let mut total_expressions = Vec::new();
    for semi_result in &semi_results {
        let full_expressions =
            reconstruct_full_expression_all(egraph, root_id, &semi_result.choices);
        total_expressions.extend(full_expressions);
    }

    total_expressions
}

pub fn list_expressions_with_target_cost_v3(
    runner: &egg::Runner<TileLang, LoopAnalysis>,
) -> Vec<String> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0];

    let max_cost = 3;
    let max_num_kernel = 1;

    let semi_results =
        enumerate_expressions_with_target_cost_v2(egraph, root_id, max_cost, max_num_kernel);

    println!("There are {} semi-expressions", &semi_results.len());

    let mut total_expressions = Vec::new();
    for semi_result in &semi_results {
        let full_expressions =
            reconstruct_full_expression_naive(egraph, root_id, &semi_result.choices);
        total_expressions.extend(full_expressions);
    }

    total_expressions
}

pub fn list_expressions_with_target_cost_v3_part1(
    runner: &egg::Runner<TileLang, LoopAnalysis>,
    output_file: &str,
    max_cost: usize,
    max_num_kernel: usize,
) -> io::Result<usize> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0];

    let semi_results =
        enumerate_expressions_with_target_cost_v2(egraph, root_id, max_cost, max_num_kernel);

    println!("There are {} semi-expressions", &semi_results.len());

    // Convert to JSON array
    let mut json_array = JsonValue::new_array();
    for result in &semi_results {
        json_array.push(result.to_json()).unwrap();
    }

    let json_data = object! {
        count: semi_results.len(),
        semi_expressions: json_array
    };

    // Convert to pretty-printed string
    let json_string = json::stringify_pretty(json_data, 2);

    // Write to file
    fs::write(output_file, json_string)?;

    println!("Semi-expressions saved to {}", output_file);
    Ok(semi_results.len())
}

pub fn list_expressions_with_target_cost_v3_part2(
    runner: &egg::Runner<TileLang, LoopAnalysis>,
    input_file: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0];

    // Read from file
    let json_string = fs::read_to_string(input_file)?;

    // Parse JSON
    let parsed = json::parse(&json_string).map_err(|e| format!("JSON parse error: {}", e))?;

    let mut semi_results = Vec::new();

    // Extract semi_expressions array
    if parsed["semi_expressions"].is_array() {
        let expressions_array = &parsed["semi_expressions"];
        for i in 0..expressions_array.len() {
            let expression_json = &expressions_array[i];
            match SemiExpressionResult::from_json(expression_json) {
                Ok(result) => semi_results.push(result),
                Err(e) => eprintln!("Warning: Failed to parse expression: {}", e),
            }
        }
    } else {
        return Err("Missing or invalid 'semi_expressions' array in JSON".into());
    }

    println!(
        "Loaded {} semi-expressions from {}",
        semi_results.len(),
        input_file
    );

    let mut total_expressions = Vec::new();
    for semi_result in &semi_results {
        let full_expressions =
            reconstruct_full_expression_naive(egraph, root_id, &semi_result.choices);
        total_expressions.extend(full_expressions);
    }

    Ok(total_expressions)
}

pub fn list_expressions_from_semi_all(
    runner: &egg::Runner<TileLang, LoopAnalysis>,
    input_file: &str,
    index: usize,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0];

    // Read from file
    let json_string = fs::read_to_string(input_file)?;

    // Parse JSON
    let parsed = json::parse(&json_string).map_err(|e| format!("JSON parse error: {}", e))?;

    let mut semi_results = Vec::new();

    // Extract semi_expressions array
    if parsed["semi_expressions"].is_array() {
        let expressions_array = &parsed["semi_expressions"];
        for i in 0..expressions_array.len() {
            let expression_json = &expressions_array[i];
            match SemiExpressionResult::from_json(expression_json) {
                Ok(result) => semi_results.push(result),
                Err(e) => eprintln!("Warning: Failed to parse expression: {}", e),
            }
        }
    } else {
        return Err("Missing or invalid 'semi_expressions' array in JSON".into());
    }

    println!(
        "Loaded {} semi-expressions from {}",
        semi_results.len(),
        input_file
    );

    let semi_result = &semi_results[index];
    let full_expressions = reconstruct_full_expression_all(egraph, root_id, &semi_result.choices);
    Ok(full_expressions)
}

pub fn list_expressions_from_semi_naive(
    runner: &egg::Runner<TileLang, LoopAnalysis>,
    input_file: &str,
    index: usize,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0];

    // Read from file
    let json_string = fs::read_to_string(input_file)?;

    // Parse JSON
    let parsed = json::parse(&json_string)?;

    let mut semi_results = Vec::new();

    // Extract semi_expressions array
    if parsed["semi_expressions"].is_array() {
        let expressions_array = &parsed["semi_expressions"];
        for i in 0..expressions_array.len() {
            let expression_json = &expressions_array[i];
            match SemiExpressionResult::from_json(expression_json) {
                Ok(result) => semi_results.push(result),
                Err(e) => eprintln!("Warning: Failed to parse expression: {}", e),
            }
        }
    } else {
        return Err("Missing or invalid 'semi_expressions' array in JSON".into());
    }

    println!(
        "Loaded {} semi-expressions from {}",
        semi_results.len(),
        input_file
    );

    if index >= semi_results.len() {
        return Err(format!(
            "Index {} out of range. Only {} semi-expressions available",
            index,
            semi_results.len()
        )
        .into());
    }

    let semi_result = &semi_results[index];
    let (new_egraph, new_root_id) =
        create_egraph_from_semi_expression(egraph, semi_result, root_id);
    let full_expressions = reconstruct_full_expression_naive(&new_egraph, new_root_id, &[]);
    println!("{:?}", semi_result.semi_expression);
    println!("{:?}", full_expressions);
    // let dot_string = new_egraph.dot().to_string();
    // let mut file = File::create("egraph.dot")?;
    // file.write_all(dot_string.as_bytes())?;
    // println!("EGraph visualization saved to egraph.dot");
    Ok(full_expressions)
}

pub fn list_expressions_from_semi_with_cost(
    runner: &egg::Runner<TileLang, LoopAnalysis>,
    input_file: &str,
    index: usize,
) -> Result<(Vec<String>, Vec<HashSet<String>>), Box<dyn std::error::Error>> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0];

    // Read from file
    let json_string = fs::read_to_string(input_file)?;

    // Parse JSON
    let parsed = json::parse(&json_string)?;

    let mut semi_results = Vec::new();

    // Extract semi_expressions array
    if parsed["semi_expressions"].is_array() {
        let expressions_array = &parsed["semi_expressions"];
        for i in 0..expressions_array.len() {
            let expression_json = &expressions_array[i];
            match SemiExpressionResult::from_json(expression_json) {
                Ok(result) => semi_results.push(result),
                Err(e) => eprintln!("Warning: Failed to parse expression: {}", e),
            }
        }
    } else {
        return Err("Missing or invalid 'semi_expressions' array in JSON".into());
    }

    println!(
        "Loaded {} semi-expressions from {}",
        semi_results.len(),
        input_file
    );

    // Check if index is special value (usize::MAX) to return all best expressions
    if index == usize::MAX {
        let mut multi_enode_tile_sets = Vec::new();

        // Collect multi-enode tile information from the first semi-expression only
        if !semi_results.is_empty() {
            let (first_egraph, _) =
                create_egraph_from_semi_expression(egraph, &semi_results[0], root_id);

            // Detect multi-enode tile eclasses
            for class in first_egraph.classes() {
                // Count Tile and Elem nodes
                let tile_elem_nodes: Vec<_> = class
                    .nodes
                    .iter()
                    .filter(|node| matches!(node, TileLang::Tile(_) | TileLang::Elem(_)))
                    .collect();
                // println!("{:?}", tile_elem_nodes);
                // Only process if there are multiple Tile/Elem nodes
                if tile_elem_nodes.len() > 1 {
                    let mut tile_elem_exprs = HashSet::new();

                    for node in tile_elem_nodes {
                        match node {
                            TileLang::Tile(var_id) => {
                                let var_names = get_all_var_strings(&first_egraph, *var_id);
                                for var_name in var_names {
                                    // Skip dummydata variables
                                    if !var_name.starts_with("dummydata") {
                                        tile_elem_exprs.insert(format!("(tile {})", var_name));
                                    }
                                }
                            }
                            TileLang::Elem(var_id) => {
                                let var_names = get_all_var_strings(&first_egraph, *var_id);
                                for var_name in var_names {
                                    // Skip dummydata variables
                                    if !var_name.starts_with("dummydata") {
                                        tile_elem_exprs.insert(format!("(elem {})", var_name));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    // Check if they have different variables (should be true if we have multiple nodes)
                    if tile_elem_exprs.len() > 1 {
                        multi_enode_tile_sets.push(tile_elem_exprs);
                    }
                }
            }
        }

        // Now extract best expressions for all semi-expressions

        // Original sequential version (commented out for parallel implementation)
        // for (i, semi_result) in semi_results.iter().enumerate() {
        //     let (new_egraph, new_root_id) = create_egraph_from_semi_expression(egraph, semi_result, root_id);
        //     let extractor = create_fine_grained_extractor(&new_egraph);
        //     // let extractor = Extractor::new(&new_egraph, AstSize);
        //     // let extractor = Extractor::new(&egraph, AstSize);
        //     // let (best_cost, best_expr) = extractor.find_best(root_id);
        //     let (best_cost, best_expr) = extractor.find_best(new_root_id);

        //     // all_best_expressions.push(format!("// Semi-expression index {}: cost = {}", i, best_cost));
        //     all_best_expressions.push(format!("{}", best_expr));
        // }

        // Parallel version using rayon
        let best_expressions: Vec<String> = semi_results
            .par_iter()
            .enumerate()
            .map(|(_i, semi_result)| {
                let (new_egraph, new_root_id) =
                    create_egraph_from_semi_expression(egraph, semi_result, root_id);
                let extractor = create_fine_grained_extractor(&new_egraph);
                // let extractor = Extractor::new(&new_egraph, AstSize);
                // let extractor = Extractor::new(&egraph, AstSize);
                // let (best_cost, best_expr) = extractor.find_best(root_id);
                let (_best_cost, best_expr) = extractor.find_best(new_root_id);
                // println!("semi {:?} cost {:?}", i, best_cost);

                // format!("// Semi-expression index {}: cost = {}", i, best_cost)
                format!("{}", best_expr)
            })
            .collect();

        let all_best_expressions = best_expressions;

        println!(
            "Extracted best expressions for all {} semi-expressions",
            semi_results.len()
        );
        return Ok((all_best_expressions, multi_enode_tile_sets));
    }

    if index >= semi_results.len() {
        return Err(format!(
            "Index {} out of range. Only {} semi-expressions available",
            index,
            semi_results.len()
        )
        .into());
    }

    let semi_result = &semi_results[index];
    let (new_egraph, new_root_id) =
        create_egraph_from_semi_expression(egraph, semi_result, root_id);

    // Collect multi-enode tile information
    let mut multi_enode_tile_sets = Vec::new();
    for class in new_egraph.classes() {
        // Count Tile and Elem nodes
        let tile_elem_nodes: Vec<_> = class
            .nodes
            .iter()
            .filter(|node| matches!(node, TileLang::Tile(_) | TileLang::Elem(_)))
            .collect();

        // Only process if there are multiple Tile/Elem nodes
        if tile_elem_nodes.len() > 1 {
            let mut tile_elem_exprs = HashSet::new();

            for node in tile_elem_nodes {
                match node {
                    TileLang::Tile(var_id) => {
                        if let Some(var_name) = get_var_string(&new_egraph, *var_id) {
                            tile_elem_exprs.insert(format!("(tile {})", var_name));
                        }
                    }
                    TileLang::Elem(var_id) => {
                        if let Some(var_name) = get_var_string(&new_egraph, *var_id) {
                            tile_elem_exprs.insert(format!("(elem {})", var_name));
                        }
                    }
                    _ => {}
                }
            }

            // Check if they have different variables (should be true if we have multiple nodes)
            if tile_elem_exprs.len() > 1 {
                multi_enode_tile_sets.push(tile_elem_exprs);
            }
        }
    }

    let extractor = create_fine_grained_extractor(&new_egraph);
    let (_best_cost, best_expr) = extractor.find_best(new_root_id);

    Ok((vec![format!("{}", best_expr)], multi_enode_tile_sets))
}
