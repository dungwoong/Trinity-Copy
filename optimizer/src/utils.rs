//! Utility functions used across multiple modules

use crate::language::TileLang;
use crate::language::{Access, LoopAnalysis};
use egg::*;
use std::collections::{HashMap, HashSet, VecDeque};

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

pub fn collect_access_sets(
    egraph: &EGraph,
    root: Id,
    need_source: bool,
) -> (Vec<Access>, Vec<Access>) {
    let mut read_set = vec![];
    let mut write_set = vec![];
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(root);

    while let Some(id) = queue.pop_front() {
        if !visited.insert(id) {
            continue;
        }

        let data = &egraph[id].data;
        for enode in &egraph[id].nodes {
            if data.is_deleted.contains(enode) {
                continue;
            }

            match enode {
                TileLang::Load([base, idx]) => {
                    for base_node in &egraph[*base].nodes {
                        match base_node {
                            TileLang::Input(tensor_id)
                            | TileLang::Output(tensor_id)
                            | TileLang::Tensor(tensor_id) => {
                                for tensor_node in &egraph[*tensor_id].nodes {
                                    if let TileLang::Var(sym) = tensor_node {
                                        let base_name = sym.as_str().to_string();
                                        let idx_expr = extract_expr(egraph, *idx);
                                        let _src = if need_source {
                                            Some(enode.clone())
                                        } else {
                                            None
                                        };
                                        read_set.push(Access {
                                            base: Some(base_name),
                                            index: idx_expr,
                                        });
                                    }
                                }
                            }
                            // TileLang::Output(tensor_id) => {
                            //     for tensor_node in &egraph[*tensor_id].nodes {
                            //         if let TileLang::Var(sym) = tensor_node {
                            //             let base_name = sym.as_str().to_string();
                            //             let idx_expr = extract_expr(egraph, *idx);
                            //             let src = if need_source { Some(enode.clone()) } else { None };
                            //             read_set.push(Access {
                            //                 base: Some(base_name),
                            //                 index: idx_expr,
                            //             });
                            //         }
                            //     }
                            // }
                            _ => {}
                        }
                    }
                }

                TileLang::Store([base, _val, idx]) => {
                    for base_node in &egraph[*base].nodes {
                        match base_node {
                            TileLang::Input(tensor_id)
                            | TileLang::Output(tensor_id)
                            | TileLang::Tensor(tensor_id) => {
                                for tensor_node in &egraph[*tensor_id].nodes {
                                    if let TileLang::Var(sym) = tensor_node {
                                        let base_name = sym.as_str().to_string();
                                        let idx_expr = extract_expr(egraph, *idx);
                                        let _src = if need_source {
                                            Some(enode.clone())
                                        } else {
                                            None
                                        };
                                        write_set.push(Access {
                                            base: Some(base_name),
                                            index: idx_expr,
                                        });
                                    }
                                }
                            }
                            // TileLang::Output(tensor_id) => {
                            //     for tensor_node in &egraph[*tensor_id].nodes {
                            //         if let TileLang::Var(sym) = tensor_node {
                            //             let base_name = sym.as_str().to_string();
                            //             let idx_expr = extract_expr(egraph, *idx);
                            //             let src = if need_source { Some(enode.clone()) } else { None };
                            //             write_set.push(Access {
                            //                 base: Some(base_name),
                            //                 index: idx_expr,
                            //             });
                            //         }
                            //     }
                            // }
                            _ => {}
                        }
                    }
                }

                _ => {}
            }

            for &child in enode.children() {
                queue.push_back(child);
            }
        }
    }

    (read_set, write_set)
}

/// Flatten a nested seq structure into a list of elements
pub fn flatten_seq(egraph: &EGraph, id: Id, out: &mut Vec<Id>) {
    let data = &egraph[id].data;

    for node in &egraph[id].nodes {
        if data.is_deleted.contains(node) {
            continue; // Skip deleted enodes
        }
        if !is_legal_seq_node(egraph, &node) {
            continue;
        }
        match node {
            TileLang::Seq([left, right]) => {
                flatten_seq(egraph, *left, out);
                flatten_seq(egraph, *right, out);
                return; // Done for this seq path
            }
            _ => {}
        }
    }

    out.push(id); // Not a seq: treat as leaf
}

/// Checks whether a single Seq enode is a legal sequence tree
pub fn is_legal_seq_node(egraph: &EGraph, node: &TileLang) -> bool {
    if let TileLang::Seq([left, right]) = node {
        is_legal_seq_tree_strict(egraph, *left, *right)
    } else {
        false
    }
}

/// Recursively check if a given (left, right) sequence is legal
pub fn is_legal_seq_tree_strict(egraph: &EGraph, left: Id, right: Id) -> bool {
    // Left must NOT be a Seq
    let left_data = &egraph[left].data;
    let left_is_seq = egraph[left]
        .nodes
        .iter()
        .filter(|n| !left_data.is_deleted.contains(n))
        .any(|n| matches!(n, TileLang::Seq([_, _])));
    if left_is_seq {
        // println!("Left is seq!");
        return false;
    }

    // Right must be either a Seq (legal) or a leaf
    for node in &egraph[right].nodes {
        if egraph[right].data.is_deleted.contains(&node) {
            continue;
        }
        match node {
            TileLang::Seq([rleft, rright]) => {
                return is_legal_seq_tree_strict(egraph, *rleft, *rright);
            }
            _ => {} // leaf, fine
        }
    }

    true
}

// Returns all possible sequence arrangements from an e-class
pub fn flatten_seq_all_branch(egraph: &EGraph, id: Id, should_legal: bool) -> Vec<Vec<Id>> {
    let data = &egraph[id].data;
    let mut all_sequences = Vec::new();

    for node in &egraph[id].nodes {
        if data.is_deleted.contains(node) {
            continue;
        }

        if let TileLang::Seq([left, right]) = node {
            if should_legal && !is_legal_seq_node(egraph, node) {
                continue; // Skip illegal sequences
            }

            let left_seqs = flatten_seq_all_branch(egraph, *left, true);
            let right_seqs = flatten_seq_all_branch(egraph, *right, true);

            // Cartesian product of left and right sequences
            for left_seq in &left_seqs {
                for right_seq in &right_seqs {
                    let mut combined = left_seq.clone();
                    combined.extend(right_seq.clone());
                    all_sequences.push(combined);
                }
            }
        }
    }

    // If no sequences found, treat as leaf
    if all_sequences.is_empty() {
        all_sequences.push(vec![id]);
    }

    all_sequences
}

pub fn collect_reachable_eclasses(egraph: &EGraph, root: Id) -> HashSet<Id> {
    let mut visited = HashSet::default();
    let mut queue = VecDeque::new();
    queue.push_back(root);

    while let Some(id) = queue.pop_front() {
        if visited.insert(id) {
            let data = &egraph[id].data;
            for enode in &egraph[id].nodes {
                if data.is_deleted.contains(&enode) {
                    continue;
                }
                for &child in enode.children() {
                    queue.push_back(child);
                }
            }
        }
    }
    visited
}

// pub fn extract_expr(egraph: &EGraph, id: Id) -> Option<TileLang> {
//     egraph[id].nodes.iter().next().cloned()
// }
pub fn extract_expr(egraph: &EGraph, id: Id) -> Vec<TileLang> {
    egraph[id].nodes.iter().cloned().collect()
}

pub fn extract_store_info(egraph: &EGraph, id: Id) -> Option<(String, Vec<TileLang>, Id)> {
    let data = &egraph[id].data;
    for enode in &egraph[id].nodes {
        if data.is_deleted.contains(&enode) {
            continue;
        }
        if let TileLang::Store([base_id, val_id, idx_id]) = enode {
            // base_id should point to TileLang::Input(Var(sym))
            for base_node in &egraph[*base_id].nodes {
                match base_node {
                    TileLang::Input(tensor_id)
                    | TileLang::Output(tensor_id)
                    | TileLang::Tensor(tensor_id) => {
                        for tensor_node in &egraph[*tensor_id].nodes {
                            if let TileLang::Var(sym) = tensor_node {
                                let base_name = sym.as_str().to_string();
                                let idx_expr = extract_expr(egraph, *idx_id);
                                return Some((base_name, idx_expr, *val_id));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    None
}

pub fn contains_loop(egraph: &EGraph, id: Id) -> bool {
    let data = &egraph[id].data;
    egraph[id]
        .nodes
        .iter()
        .filter(|n| !data.is_deleted.contains(n))
        .any(|n| matches!(n, TileLang::Loop(_)))
}

/// Check whether there exists same access in readset and writeset
pub fn is_reduction_op(egraph: &EGraph, id: Id) -> bool {
    let (reads, writes) = collect_access_sets(egraph, id, false);

    for read in &reads {
        for write in &writes {
            if base_overlap(read, write) {
                return true;
            }
        }
    }

    false
}

pub fn get_var_symbol(egraph: &EGraph, id: Id) -> Option<&egg::Symbol> {
    for node in &egraph[id].nodes {
        if let TileLang::Var(sym) = node {
            return Some(sym);
        }
    }
    None
}

pub fn get_var_string(egraph: &EGraph, id: Id) -> Option<String> {
    let eclass = &egraph[id];
    for enode in &eclass.nodes {
        if let TileLang::Var(symbol) = enode {
            return Some(symbol.to_string());
        }
    }
    None
}

pub fn get_all_var_strings(egraph: &EGraph, id: Id) -> Vec<String> {
    let eclass = &egraph[id];
    let mut var_strings = Vec::new();
    for enode in &eclass.nodes {
        if let TileLang::Var(symbol) = enode {
            var_strings.push(symbol.to_string());
        }
    }
    var_strings
}

pub fn get_base_name(expr: &RecExpr<TileLang>, node_idx: usize) -> Option<String> {
    if node_idx >= expr.as_ref().len() {
        return None;
    }

    let node = &expr.as_ref()[node_idx];
    match node {
        TileLang::Input(tensor_id) | TileLang::Output(tensor_id) | TileLang::Tensor(tensor_id) => {
            get_var_name(expr, usize::from(*tensor_id))
        }
        TileLang::Var(symbol) => Some(symbol.to_string()),
        _ => None,
    }
}

pub fn get_var_name(expr: &RecExpr<TileLang>, node_idx: usize) -> Option<String> {
    if node_idx >= expr.as_ref().len() {
        return None;
    }

    let node = &expr.as_ref()[node_idx];
    match node {
        TileLang::Var(symbol) => Some(symbol.to_string()),
        _ => None,
    }
}

pub fn index_depends_on(index: &TileLang, egraph: &EGraph, loop_var: &str) -> bool {
    match index {
        TileLang::Index(args) => {
            args.iter().any(|id| {
                egraph[*id].nodes.iter().any(|n| match n {
                    TileLang::FullTile => false, // no dependency
                    TileLang::Tile(tile_idx) => depends_on_id(egraph, *tile_idx, loop_var),
                    TileLang::Elem(tile_idx) => depends_on_id(egraph, *tile_idx, loop_var),
                    TileLang::Index(_inner_args) => index_depends_on(n, egraph, loop_var),
                    _ => false, // not a tile structure
                })
            })
        }
        _ => false, // not an Index node
    }
}

pub fn depends_on_id(egraph: &EGraph, id: Id, loop_var: &str) -> bool {
    egraph[id]
        .nodes
        .iter()
        .any(|n| expr_depends_on(egraph, n, loop_var))
}

pub fn expr_depends_on(egraph: &EGraph, expr: &TileLang, loop_var: &str) -> bool {
    match expr {
        TileLang::Var(sym) => sym.as_str() == loop_var,
        TileLang::Num(_) => false,
        TileLang::Add([a, b])
        | TileLang::Sub([a, b])
        | TileLang::Mul([a, b])
        | TileLang::Div([a, b])
        | TileLang::Matmul([a, b]) => {
            depends_on_id(egraph, *a, loop_var) || depends_on_id(egraph, *b, loop_var)
        }
        _ => false,
    }
}

pub fn has_cross_iteration_dependency(
    write: &Access,
    read_set: &[Access],
    loop_var: &str,
    egraph: &EGraph,
) -> bool {
    for write_idx in &write.index {
        if index_depends_on(write_idx, egraph, loop_var) {
            return false;
        }
    }

    for read in read_set {
        for read_idx in &read.index {
            if index_depends_on(read_idx, egraph, loop_var) {
                return true;
            }
        }
    }
    false

    // if let Some(write_idx) = &write.index {
    //     if index_depends_on(write_idx, egraph, loop_var) {
    //         return false;
    //     }
    // }

    // for read in read_set {
    //     if let Some(read_idx) = &read.index {
    //         if index_depends_on(read_idx, egraph, loop_var) {
    //             return true;
    //         }
    //     }
    // }

    // false
}

/// Returns true if the e-class has no Seq operator
pub fn no_seq(body: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body_id = subst[body];
        let data = &egraph[body_id].data;

        !egraph[body_id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Seq([_, _])))
    }
}
/// Returns true if the e-class has no Seq operator
pub fn no_loop(body: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body_id = subst[body];
        let data = &egraph[body_id].data;

        !egraph[body_id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Loop(_)))
    }
}

pub fn is_not_num(var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let id = subst[var];
        let data = &egraph[id].data;

        !egraph[id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Num(_)))
    }
}
pub fn is_num(var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let id = subst[var];
        let data = &egraph[id].data;

        egraph[id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Num(_)))
    }
}

pub fn var(s: &str) -> Var {
    s.parse().unwrap()
}

pub fn is_not_zero(var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let id = subst[var];
        let data = &egraph[id].data;

        !egraph[id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Num(0)))
    }
}
pub fn is_not_one(var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let id = subst[var];
        let data = &egraph[id].data;

        !egraph[id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Num(1)))
    }
}

pub fn cond1(n: Var, tile: Var, m: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        // Helper to extract a Num(_) value
        pub fn get_num(egraph: &EGraph, id: Id) -> Option<i32> {
            for node in &egraph[id].nodes {
                if let TileLang::Num(val) = node {
                    return Some(*val);
                }
            }
            None
        }

        let n_val = get_num(egraph, subst[n]);
        let tile_val = get_num(egraph, subst[tile]);
        let m_val = get_num(egraph, subst[m]);

        match (n_val, tile_val, m_val) {
            (Some(n), Some(tile), Some(m)) if tile != 0 => n / tile == m,
            _ => false,
        }
    }
}
pub fn no_const(var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let id = subst[var];
        let data = &egraph[id].data;

        !egraph[id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Const(_)))
    }
}

pub fn not_same_loop(
    body2: Var,
    n: Var,
    tile_n: Var,
    loop_var: Var,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body2_id = subst[body2];
        let data = &egraph[body2_id].data;

        let expected_header = (subst[n], subst[tile_n], subst[loop_var]);

        for node in &egraph[body2_id].nodes {
            if data.is_deleted.contains(node) {
                continue; // Skip deleted nodes
            }

            if let TileLang::Loop([_, loop_n, loop_tile_n, loop_var_id, _body]) = node {
                let header = (*loop_n, *loop_tile_n, *loop_var_id);
                if header == expected_header {
                    return false;
                }
            }
        }

        true
    }
}

// Helper to extract the base symbol from Input(_) or Output(_)
pub fn get_base_name_egraph(egraph: &EGraph, id: Id) -> Option<String> {
    for node in &egraph[id].nodes {
        match node {
            TileLang::Input(base_id) | TileLang::Output(base_id) | TileLang::Tensor(base_id) => {
                for base_node in &egraph[*base_id].nodes {
                    if let TileLang::Var(sym) = base_node {
                        return Some(sym.as_str().to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

pub fn is_same_base(a_var: Var, b_var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let a_id = subst[a_var];
        let b_id = subst[b_var];

        let a_base = get_base_name_egraph(egraph, a_id);
        let b_base = get_base_name_egraph(egraph, b_id);

        match (a_base, b_base) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

pub fn print_eclass(egraph: &EGraph, id: Id) {
    let class = &egraph[id];
    let data = &egraph[id].data;
    println!("EClass {} has {} enodes:", id, class.nodes.len());
    for node in &class.nodes {
        if data.is_deleted.contains(node) {
            continue;
        }
        println!("  {}", node);
    }
}

pub fn base_overlap(access1: &Access, access2: &Access) -> bool {
    match (&access1.base, &access2.base) {
        (Some(base1), Some(base2)) => bases_overlap(base1, base2),
        _ => false,
    }
}

// Check if two comma-separated base name strings have any overlap
pub fn bases_overlap(base1: &str, base2: &str) -> bool {
    let bases1: Vec<&str> = base1.split(',').map(|s| s.trim()).collect();
    let bases2: Vec<&str> = base2.split(',').map(|s| s.trim()).collect();

    for b1 in &bases1 {
        for b2 in &bases2 {
            if b1 == b2 {
                return true;
            }
        }
    }
    false
}

pub fn indices_are_same(egraph: &EGraph, indices1: &[TileLang], indices2: &[TileLang]) -> bool {
    if indices1.len() != indices2.len() {
        return false;
    }

    for (idx1, idx2) in indices1.iter().zip(indices2) {
        if !index_expressions_equal(egraph, idx1, idx2) {
            return false;
        }
    }

    true
}

fn index_expressions_equal(egraph: &EGraph, expr1: &TileLang, expr2: &TileLang) -> bool {
    // Return true if either expression is ConstTile
    if matches!(expr1, TileLang::ConstTile(_)) || matches!(expr2, TileLang::ConstTile(_)) {
        return true;
    }

    match (expr1, expr2) {
        (TileLang::Num(n1), TileLang::Num(n2)) => n1 == n2,
        (TileLang::Var(v1), TileLang::Var(v2)) => v1 == v2,
        (TileLang::Add([l1, r1]), TileLang::Add([l2, r2]))
        | (TileLang::Mul([l1, r1]), TileLang::Mul([l2, r2])) => {
            let l1_nodes = &egraph[*l1].nodes;
            let r1_nodes = &egraph[*r1].nodes;
            let l2_nodes = &egraph[*l2].nodes;
            let r2_nodes = &egraph[*r2].nodes;

            l1_nodes.iter().any(|n1| {
                l2_nodes
                    .iter()
                    .any(|n2| index_expressions_equal(egraph, n1, n2))
            }) && r1_nodes.iter().any(|n1| {
                r2_nodes
                    .iter()
                    .any(|n2| index_expressions_equal(egraph, n1, n2))
            })
        }
        (TileLang::Sub([l1, r1]), TileLang::Sub([l2, r2]))
        | (TileLang::Div([l1, r1]), TileLang::Div([l2, r2])) => {
            let l1_nodes = &egraph[*l1].nodes;
            let r1_nodes = &egraph[*r1].nodes;
            let l2_nodes = &egraph[*l2].nodes;
            let r2_nodes = &egraph[*r2].nodes;

            l1_nodes.iter().any(|n1| {
                l2_nodes
                    .iter()
                    .any(|n2| index_expressions_equal(egraph, n1, n2))
            }) && r1_nodes.iter().any(|n1| {
                r2_nodes
                    .iter()
                    .any(|n2| index_expressions_equal(egraph, n1, n2))
            })
        }
        (TileLang::Index(args1), TileLang::Index(args2)) => {
            if args1.len() != args2.len() {
                return false;
            }
            args1.iter().zip(args2.iter()).all(|(id1, id2)| {
                let nodes1 = &egraph[*id1].nodes;
                let nodes2 = &egraph[*id2].nodes;
                nodes1.iter().any(|n1| {
                    nodes2
                        .iter()
                        .any(|n2| index_expressions_equal(egraph, n1, n2))
                })
            })
        }
        (TileLang::Tile(id1), TileLang::Tile(id2)) | (TileLang::Elem(id1), TileLang::Elem(id2)) => {
            let nodes1 = &egraph[*id1].nodes;
            let nodes2 = &egraph[*id2].nodes;
            nodes1.iter().any(|n1| {
                nodes2
                    .iter()
                    .any(|n2| index_expressions_equal(egraph, n1, n2))
            })
        }
        (TileLang::FullTile, TileLang::FullTile) => true,
        (TileLang::ConstTile(c1), TileLang::ConstTile(c2)) => c1 == c2,
        _ => true,
    }
}

pub fn get_index_node(expr: &RecExpr<TileLang>, node_idx: usize) -> Option<TileLang> {
    if node_idx >= expr.as_ref().len() {
        return None;
    }

    Some(expr.as_ref()[node_idx].clone())
}

pub fn is_output_base(expr: &RecExpr<TileLang>, node_idx: usize) -> bool {
    if node_idx >= expr.as_ref().len() {
        return false;
    }

    let node = &expr.as_ref()[node_idx];
    matches!(node, TileLang::Output(_))
}

pub fn measure_enode_proportions(egraph: &EGraph, print: bool) -> HashMap<String, (usize, f64)> {
    let mut node_counts: HashMap<String, usize> = HashMap::new();
    let mut total_nodes = 0usize;

    // Count all nodes across all eclasses
    for eclass in egraph.classes() {
        for enode in &eclass.nodes {
            let node_type = get_enode_type_name(enode);
            *node_counts.entry(node_type).or_insert(0) += 1;
            total_nodes += 1;
        }
    }

    // Calculate proportions
    let mut proportions: HashMap<String, (usize, f64)> = HashMap::new();
    for (node_type, count) in node_counts {
        let proportion = if total_nodes > 0 {
            count as f64 / total_nodes as f64
        } else {
            0.0
        };
        proportions.insert(node_type, (count, proportion));
    }

    if print {
        let mut sorted_proportions: Vec<_> = proportions.clone().into_iter().collect();
        sorted_proportions.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

        println!("{:<15} {:<8} {:<10}", "Type", "Count", "Proportion");
        println!("{}", "-".repeat(35));
        for (node_type, (count, proportion)) in sorted_proportions.iter().take(10) {
            println!(
                "{}: {} nodes ({:.2}%)",
                node_type,
                count,
                proportion * 100.0
            );
        }
    }

    proportions
}

fn get_enode_type_name(enode: &TileLang) -> String {
    match enode {
        TileLang::Loop(_) => "Loop".to_string(),
        TileLang::DLoop(_) => "DLoop".to_string(),
        TileLang::TLoop(_) => "TLoop".to_string(),
        TileLang::Input(_) => "Input".to_string(),
        TileLang::Output(_) => "Output".to_string(),
        TileLang::Tensor(_) => "Tensor".to_string(),
        TileLang::Tile(_) => "Tile".to_string(),
        TileLang::FullTile => "FullTile".to_string(),
        TileLang::Elem(_) => "Elem".to_string(),
        TileLang::ConstTile(_) => "ConstTile".to_string(),
        TileLang::Index(_) => "Index".to_string(),
        TileLang::Load(_) => "Load".to_string(),
        TileLang::Store(_) => "Store".to_string(),
        TileLang::Seq(_) => "Seq".to_string(),
        TileLang::Const(_) => "Const".to_string(),
        TileLang::Add(_) => "Add".to_string(),
        TileLang::Sub(_) => "Sub".to_string(),
        TileLang::Mul(_) => "Mul".to_string(),
        TileLang::Div(_) => "Div".to_string(),
        TileLang::Le(_) => "Le".to_string(),
        TileLang::Max(_) => "Max".to_string(),
        TileLang::Min(_) => "Min".to_string(),
        TileLang::Exp(_) => "Exp".to_string(),
        TileLang::Erf(_) => "Erf".to_string(),
        TileLang::Abs(_) => "Abs".to_string(),
        TileLang::Cast(_) => "Cast".to_string(),
        TileLang::Matmul(_) => "Matmul".to_string(),
        TileLang::ReduceSum(_) => "ReduceSum".to_string(),
        TileLang::ReduceMin(_) => "ReduceMin".to_string(),
        TileLang::ReduceMax(_) => "ReduceMax".to_string(),
        TileLang::Concat(_) => "Concat".to_string(),
        TileLang::Broadcast(_) => "Broadcast".to_string(),
        TileLang::Permute3(_) => "Permute3".to_string(),
        TileLang::Permute4(_) => "Permute4".to_string(),
        TileLang::Squeeze(_) => "Squeeze".to_string(),
        TileLang::Unsqueeze(_) => "Unsqueeze".to_string(),
        TileLang::Dummy => "Dummy".to_string(),
        TileLang::SLoop(_) => "SLoop".to_string(),
        TileLang::PLoop(_) => "PLoop".to_string(),
        TileLang::Num(_) => "Num".to_string(),
        TileLang::Var(_) => "Var".to_string(),
        _ => "Others".to_string(),
    }
}

// EGraph version of loop carried dependency analysis
pub fn has_loop_carried_dependency_egraph(egraph: &EGraph, body_id: Id, loop_var_id: Id) -> bool {
    let loop_var_name = egraph[loop_var_id]
        .nodes
        .iter()
        .find_map(|n| {
            if let TileLang::Var(sym) = n {
                Some(sym.as_str().to_string())
            } else {
                None
            }
        })
        .expect("loop_var should be a Var");

    // 1. collect all write access from the body
    // let mut read_accesses = Vec::new();
    // let mut write_accesses = Vec::new();
    // collect_access_sets(egraph, body_id, &mut read_accesses, &mut write_accesses);

    let (_read_accesses, write_accesses) = collect_access_sets(egraph, body_id, false);

    // 2. check whether the index of write access contains loop_var or not.
    for write_access in &write_accesses {
        if !write_access.index.is_empty() {
            let all_independent = write_access
                .index
                .iter()
                .all(|index| !index_depends_on(index, egraph, &loop_var_name));

            if all_independent {
                // 3. if ALL indices don't contain loop_var, return true (meaning has dependency -> sloop)
                return true;
            }
        }
        // if let Some(index) = &write_access.index {
        //     if !index_depends_on(index, egraph, &loop_var_name) {
        //         // 3. if not contain loop_var, return true (meaning has dependency -> sloop)
        //         return true;
        //     }
        // }
    }

    // if all write accesses contain loop_var, return false (meaning no dependency -> ploop)
    false
}
