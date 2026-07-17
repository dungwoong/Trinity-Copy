//! Postprocessing and deadcode elimination

use crate::language::*;
use crate::utils::*;
use egg::*;
use std::collections::HashMap;
use std::collections::HashSet;

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

pub fn postprocess_egraph(egraph: &mut EGraph) {
    /*
    Step1) Finally resolve all illegal sequences
    */
    let class_ids: Vec<Id> = egraph.classes().map(|class| class.id).collect();

    for id in class_ids {
        let nodes = egraph[id].nodes.clone();
        for node in nodes {
            if is_legal_seq_node(egraph, &node) {
                continue; // there already exists legal sequence node
            }

            let mut new_forms = vec![];
            if let TileLang::Seq([left, right]) = node {
                let mut seq_elements = vec![];

                // Step 1: Flatten recursively
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

                // egraph[id].data.is_deleted.insert(node);
            }
        }
    }

    let class_ids: Vec<Id> = egraph.classes().map(|class| class.id).collect();
    for id in class_ids {
        let nodes = egraph[id].nodes.clone();
        let mut legal = false;
        for node in nodes {
            if let TileLang::Seq([_left, _right]) = node {
                if is_legal_seq_node(egraph, &node) {
                    legal = true;
                }
            } else {
                legal = true;
            }
        }
        if !legal {
            println!("Still illegal, {}", id);
            print_eclass(egraph, id);
        }
    }

    egraph.rebuild();
}

pub fn postprocess(input: &str) -> RecExpr<TileLang> {
    /*
        1. Parse input AST string into RecExpr format
        2. Deadcode elimination
            2-1) From the root, traverse entire AST
            2-2) When meeting Load(base, _) operator, insert base to read_set
            2-3) Traverse the entire AST from root againg
            2-4) When meeting Store(base, _, _) operator, do the following
                if base doesn't exist in read_set and base is not Output(_) operator,
                Substitute the Store() operator to Dummy() operator
        3. Decide parallel/sequential loop
            3-1) From the root, traverse entire AST
            3-2) for all Loop() operator, do the following
                if loop body has cross iteration dependency, substitute to SLoop()
                else, substitute to PLoop()

                if loop body is Seq() operator, substitute all reachable Loop() operator from the loop body to SLoop()

    */
    let mut expr: RecExpr<TileLang> = input.parse().unwrap();

    for _ in 0..3 {
        value_forward_substitution(&mut expr);
    }
    for _ in 0..3 {
        expr = deadcode_elimination(expr);
    }
    expr = decide_loop_types(expr);
    expr
}

pub fn postprocess_v2(
    input: &str,
    multi_enode_tile_sets: &Vec<HashSet<String>>,
) -> RecExpr<TileLang> {
    // Apply loop variable coherence check and substitution on string first
    let coherent_string = apply_loop_variable_coherence_string(input, multi_enode_tile_sets);

    // Parse the coherent string into RecExpr
    let mut expr: RecExpr<TileLang> = coherent_string.parse().unwrap();

    for _ in 0..5 {
        value_forward_substitution(&mut expr);
    }
    for _ in 0..3 {
        expr = deadcode_elimination(expr);
    }
    expr = decide_loop_types(expr);
    expr
}

pub fn apply_loop_variable_coherence_string(
    input: &str,
    multi_enode_tile_sets: &Vec<HashSet<String>>,
) -> String {
    let mut tile_equivalences: HashMap<String, Vec<String>> = HashMap::new();

    for tile_set in multi_enode_tile_sets {
        let tiles: Vec<_> = tile_set.iter().cloned().collect();
        for tile in &tiles {
            tile_equivalences.insert(tile.clone(), tiles.clone());
        }
    }

    // Parse and apply coherence
    let mut result = input.to_string();
    let loop_scopes = find_loop_scopes(&result);
    let tile_elem_positions = find_tile_elem_expressions(&result);

    // Apply substitutions from right to left to avoid position shifts
    let mut substitutions: Vec<(usize, usize, String)> = Vec::new();

    for (start_pos, end_pos, expr_str) in tile_elem_positions.iter().rev() {
        if let Some((_op_type, var_name)) = parse_tile_elem_expr(expr_str) {
            // Find which loops are in scope at this position
            let in_scope_vars = get_in_scope_vars(&loop_scopes, *start_pos);

            // Check if current variable is in scope
            if !in_scope_vars.contains(&var_name) {
                // Try to find an equivalent expression with in-scope variable
                if let Some(equivalent_exprs) = tile_equivalences.get(expr_str) {
                    for equiv_expr in equivalent_exprs {
                        if let Some((_, equiv_var)) = parse_tile_elem_expr(equiv_expr) {
                            if in_scope_vars.contains(&equiv_var) {
                                // Found a valid substitution
                                substitutions.push((*start_pos, *end_pos, equiv_expr.clone()));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply all substitutions
    for (start, end, new_expr) in substitutions.iter().rev() {
        result.replace_range(*start..*end, new_expr);
    }

    result
}

// Helper structure to represent a loop scope
struct LoopScope {
    start_pos: usize,
    end_pos: usize,
    loop_var: String,
}

// Helper function to find all loop scopes in the expression
fn find_loop_scopes(expr: &str) -> Vec<LoopScope> {
    let mut scopes = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if i + 5 < chars.len() && &chars[i..i + 5] == ['(', 'l', 'o', 'o', 'p'] {
            if let Some(scope) = parse_loop_at_position(&chars, i) {
                scopes.push(scope);
            }
        }
        i += 1;
    }

    scopes
}

// Helper function to find all tile/elem expressions
fn find_tile_elem_expressions(expr: &str) -> Vec<(usize, usize, String)> {
    let mut expressions = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if i + 5 < chars.len() && &chars[i..i + 5] == ['(', 't', 'i', 'l', 'e'] {
            if let Some((end_pos, expr_str)) = extract_expression(&chars, i) {
                expressions.push((i, end_pos, expr_str));
            }
        } else if i + 5 < chars.len() && &chars[i..i + 5] == ['(', 'e', 'l', 'e', 'm'] {
            if let Some((end_pos, expr_str)) = extract_expression(&chars, i) {
                expressions.push((i, end_pos, expr_str));
            }
        }
        i += 1;
    }

    expressions
}

// Parse a loop at a given position
fn parse_loop_at_position(chars: &[char], start: usize) -> Option<LoopScope> {
    // Find the matching closing parenthesis
    let mut paren_count = 0;
    let mut i = start;
    let mut end_pos = start;

    while i < chars.len() {
        if chars[i] == '(' {
            paren_count += 1;
        } else if chars[i] == ')' {
            paren_count -= 1;
            if paren_count == 0 {
                end_pos = i + 1;
                break;
            }
        }
        i += 1;
    }

    if end_pos == start {
        return None;
    }

    // Extract the loop expression
    let loop_str: String = chars[start..end_pos].iter().collect();

    // Parse the loop variable (4th argument)
    let parts: Vec<&str> = loop_str[6..loop_str.len() - 1].split_whitespace().collect();
    if parts.len() >= 4 {
        Some(LoopScope {
            start_pos: start,
            end_pos,
            loop_var: parts[3].to_string(),
        })
    } else {
        None
    }
}

// Extract a complete expression starting at position
fn extract_expression(chars: &[char], start: usize) -> Option<(usize, String)> {
    let mut paren_count = 0;
    let mut i = start;

    while i < chars.len() {
        if chars[i] == '(' {
            paren_count += 1;
        } else if chars[i] == ')' {
            paren_count -= 1;
            if paren_count == 0 {
                let expr: String = chars[start..=i].iter().collect();
                return Some((i + 1, expr));
            }
        }
        i += 1;
    }

    None
}

// Get variables that are in scope at a given position
fn get_in_scope_vars(scopes: &[LoopScope], position: usize) -> HashSet<String> {
    let mut in_scope = HashSet::new();

    for scope in scopes {
        if position >= scope.start_pos && position < scope.end_pos {
            in_scope.insert(scope.loop_var.clone());
        }
    }

    in_scope
}

fn parse_tile_elem_expr(expr_str: &str) -> Option<(&str, String)> {
    if expr_str.starts_with("(tile ") && expr_str.ends_with(")") {
        let var = expr_str[6..expr_str.len() - 1].to_string();
        Some(("tile", var))
    } else if expr_str.starts_with("(elem ") && expr_str.ends_with(")") {
        let var = expr_str[6..expr_str.len() - 1].to_string();
        Some(("elem", var))
    } else {
        None
    }
}

pub fn deadcode_elimination(mut expr: RecExpr<TileLang>) -> RecExpr<TileLang> {
    let mut read_set = HashSet::new();
    let root_idx = expr.as_ref().len() - 1;
    collect_read_set(&expr, root_idx, &mut read_set);

    eliminate_dead_stores(&mut expr, root_idx, &read_set);

    expr
}

pub fn collect_read_set(expr: &RecExpr<TileLang>, node_idx: usize, read_set: &mut HashSet<String>) {
    if node_idx >= expr.as_ref().len() {
        return;
    }

    let node = &expr.as_ref()[node_idx];
    match node {
        TileLang::Load([base_id, _]) => {
            if let Some(base_name) = get_base_name(expr, usize::from(*base_id)) {
                read_set.insert(base_name);
            }
        }
        _ => {
            for &child_id in node.children() {
                collect_read_set(expr, usize::from(child_id), read_set);
            }
        }
    }
}

pub fn eliminate_dead_stores(
    expr: &mut RecExpr<TileLang>,
    node_idx: usize,
    read_set: &HashSet<String>,
) {
    if node_idx >= expr.as_ref().len() {
        return;
    }

    let node = expr.as_ref()[node_idx].clone();

    match node {
        TileLang::Store([base_id, _val_id, _idx_id]) => {
            if let Some(base_name) = get_base_name(expr, usize::from(base_id)) {
                // Check if any basename overlaps with any element in read_set
                let is_read = read_set
                    .iter()
                    .any(|read_base| bases_overlap(&base_name, read_base));
                let is_output = is_output_base(expr, usize::from(base_id));

                if !is_read && !is_output {
                    expr.as_mut()[node_idx] = TileLang::Dummy;
                    return;
                }
            }
        }
        _ => {}
    }

    for &child_id in node.children() {
        eliminate_dead_stores(expr, usize::from(child_id), read_set);
    }
}

pub fn decide_loop_types(mut expr: RecExpr<TileLang>) -> RecExpr<TileLang> {
    let root_idx = expr.as_ref().len() - 1;
    decide_loop_types_recursive(&mut expr, root_idx, false);
    expr
}

pub fn decide_loop_types_recursive(
    expr: &mut RecExpr<TileLang>,
    node_idx: usize,
    force_sequential: bool,
) {
    if node_idx >= expr.as_ref().len() {
        return;
    }

    let node = expr.as_ref()[node_idx].clone();

    match node {
        TileLang::Loop([start, end, tile, loop_var, body]) => {
            let has_loop_carried_dep =
                has_loop_carried_dependency(expr, usize::from(body), usize::from(loop_var));

            let body_has_seq = contains_seq_operator(expr, usize::from(body));

            if has_loop_carried_dep || force_sequential {
                expr.as_mut()[node_idx] = TileLang::SLoop([start, end, tile, loop_var, body]);
            } else {
                expr.as_mut()[node_idx] = TileLang::PLoop([start, end, tile, loop_var, body]);
            }

            if body_has_seq {
                decide_loop_types_recursive(expr, usize::from(body), true);
            } else {
                decide_loop_types_recursive(expr, usize::from(body), false);
            }
        }
        _ => {
            for &child_id in node.children() {
                decide_loop_types_recursive(expr, usize::from(child_id), force_sequential);
            }
        }
    }
}

pub fn contains_seq_operator(expr: &RecExpr<TileLang>, node_idx: usize) -> bool {
    if node_idx >= expr.as_ref().len() {
        return false;
    }

    let node = &expr.as_ref()[node_idx];

    match node {
        TileLang::Seq(_) => true,
        _ => {
            // Recursively check children
            for &child_id in node.children() {
                if contains_seq_operator(expr, usize::from(child_id)) {
                    return true;
                }
            }
            false
        }
    }
}

// Helper function to collect store operations with their values
pub fn collect_store_operations(expr: &RecExpr<TileLang>, node_idx: usize) -> Vec<(Access, usize)> {
    let mut stores = Vec::new();
    collect_store_operations_recursive(expr, node_idx, &mut stores);
    stores
}

fn collect_store_operations_recursive(
    expr: &RecExpr<TileLang>,
    node_idx: usize,
    stores: &mut Vec<(Access, usize)>,
) {
    if node_idx >= expr.as_ref().len() {
        return;
    }

    let node = &expr.as_ref()[node_idx];
    match node {
        TileLang::Store([base_id, val_id, idx_id]) => {
            let base = get_base_name(expr, usize::from(*base_id));
            let index_node = get_index_node(expr, usize::from(*idx_id));
            let index = if let Some(idx) = index_node {
                vec![idx]
            } else {
                vec![]
            };
            stores.push((Access { base, index }, usize::from(*val_id)));
        }
        _ => {
            for &child_id in node.children() {
                collect_store_operations_recursive(expr, usize::from(child_id), stores);
            }
        }
    }
}

// Helper function to check if a subtree contains any read accesses that depend on loop_var
pub fn value_depends_on_loop_var(expr: &RecExpr<TileLang>, val_idx: usize, loop_var: &str) -> bool {
    if val_idx >= expr.as_ref().len() {
        return false;
    }

    let node = &expr.as_ref()[val_idx];

    // Direct dependency check
    if expr_depends_on_recexpr(expr, node, loop_var) {
        return true;
    }

    // Check for Load operations in the value expression
    match node {
        TileLang::Load([_, idx_id]) => {
            // Check if the load index depends on loop_var
            let index_node = get_index_node(expr, usize::from(*idx_id));
            if let Some(idx) = index_node {
                return index_involves_loop_var(expr, &idx, loop_var);
            }
        }
        _ => {
            // Recursively check children
            for &child_id in node.children() {
                if value_depends_on_loop_var(expr, usize::from(child_id), loop_var) {
                    return true;
                }
            }
        }
    }

    false
}

pub fn has_loop_carried_dependency(
    expr: &RecExpr<TileLang>,
    body_idx: usize,
    loop_var_idx: usize,
) -> bool {
    let loop_var_name = get_var_name(expr, loop_var_idx);
    if loop_var_name.is_none() {
        return false;
    }

    let loop_var = loop_var_name.as_ref().unwrap();

    // Collect all store operations with their values
    let store_operations = collect_store_operations(expr, body_idx);

    // Check each store operation
    for (write_access, val_idx) in store_operations {
        if !write_access.index.is_empty() {
            let all_indices_independent = write_access
                .index
                .iter()
                .all(|index| !index_involves_loop_var(expr, index, loop_var));

            if all_indices_independent {
                // The write index doesn't depend on loop_var
                // Now check if the value being stored depends on loop_var
                if value_depends_on_loop_var(expr, val_idx, loop_var) {
                    // Write is loop-independent but reads are loop-dependent
                    // This means there IS a loop-carried dependency
                    return true;
                }
                // If the value also doesn't depend on loop_var, then both reads and writes
                // are loop-independent, so no loop-carried dependency for this store
            }
        }
    }

    // if all write accesses either:
    // 1. contain loop_var in their indices, or
    // 2. have values that depend on loop_var
    // then there's no loop-carried dependency -> ploop
    false
}

pub fn involves_loop_variable_dependency(
    index1: &Option<TileLang>,
    index2: &Option<TileLang>,
    loop_var: &str,
    expr: &RecExpr<TileLang>,
) -> bool {
    match (index1, index2) {
        (Some(idx1), Some(idx2)) => {
            // Check if either index involves the loop variable in a dependency-creating way
            index_involves_loop_var(expr, idx1, loop_var)
                && index_involves_loop_var(expr, idx2, loop_var)
        }
        _ => false,
    }
}

pub fn index_involves_loop_var(expr: &RecExpr<TileLang>, index: &TileLang, loop_var: &str) -> bool {
    match index {
        TileLang::Index(args) => {
            args.iter().any(|id| {
                let node_idx = usize::from(*id);
                if node_idx < expr.as_ref().len() {
                    let node = &expr.as_ref()[node_idx];
                    match node {
                        TileLang::FullTile => false, // no dependency
                        TileLang::Tile(tile_idx) => {
                            depends_on_id_recexpr(expr, *tile_idx, loop_var)
                        }
                        TileLang::Elem(tile_idx) => {
                            depends_on_id_recexpr(expr, *tile_idx, loop_var)
                        }
                        TileLang::Index(_) => index_involves_loop_var(expr, node, loop_var),
                        _ => false, // not a tile structure
                    }
                } else {
                    false
                }
            })
        }
        _ => false, // not an Index node
    }
}

pub fn expr_depends_on_recexpr(expr: &RecExpr<TileLang>, node: &TileLang, loop_var: &str) -> bool {
    match node {
        TileLang::Var(sym) => sym.as_str() == loop_var,
        TileLang::Num(_) => false,
        TileLang::Add([a, b])
        | TileLang::Sub([a, b])
        | TileLang::Mul([a, b])
        | TileLang::Div([a, b])
        | TileLang::Le([a, b])
        | TileLang::Max([a, b])
        | TileLang::Min([a, b])
        | TileLang::Matmul([a, b]) => {
            depends_on_id_recexpr(expr, *a, loop_var) || depends_on_id_recexpr(expr, *b, loop_var)
        }
        TileLang::Exp(a)
        | TileLang::Sqr(a)
        | TileLang::Sqrt(a)
        | TileLang::Sigmoid(a)
        | TileLang::Erf(a)
        | TileLang::Abs(a) => {
            depends_on_id_recexpr(expr, *a, loop_var)
        }
        TileLang::ReduceSum([a, b])
        | TileLang::ReduceMin([a, b])
        | TileLang::ReduceMax([a, b]) => {
            depends_on_id_recexpr(expr, *a, loop_var) || depends_on_id_recexpr(expr, *b, loop_var)
        }
        TileLang::Tile(tile_idx) => depends_on_id_recexpr(expr, *tile_idx, loop_var),
        TileLang::Elem(tile_idx) => depends_on_id_recexpr(expr, *tile_idx, loop_var),
        TileLang::Index(_) => index_involves_loop_var(expr, node, loop_var),
        _ => false,
    }
}

pub fn depends_on_id_recexpr(expr: &RecExpr<TileLang>, id: Id, loop_var: &str) -> bool {
    let node_idx = usize::from(id);
    if node_idx < expr.as_ref().len() {
        let node = &expr.as_ref()[node_idx];
        expr_depends_on_recexpr(expr, node, loop_var)
    } else {
        false
    }
}

pub fn collect_memory_accesses(
    expr: &RecExpr<TileLang>,
    node_idx: usize,
    reads: &mut Vec<Access>,
    writes: &mut Vec<Access>,
) {
    if node_idx >= expr.as_ref().len() {
        return;
    }

    let node = &expr.as_ref()[node_idx];

    match node {
        TileLang::Load([base_id, idx_id]) => {
            let base = get_base_name(expr, usize::from(*base_id));
            let index_node = get_index_node(expr, usize::from(*idx_id));
            let index = if let Some(idx) = index_node {
                vec![idx] // Create vector with single element
            } else {
                vec![] // Create empty vector if no index
            };
            reads.push(Access { base, index });
        }
        TileLang::Store([base_id, _, idx_id]) => {
            let base = get_base_name(expr, usize::from(*base_id));
            let index_node = get_index_node(expr, usize::from(*idx_id));
            let index = if let Some(idx) = index_node {
                vec![idx] // Create vector with single element
            } else {
                vec![] // Create empty vector if no index
            };
            writes.push(Access { base, index });
        }
        _ => {}
    }

    for &child_id in node.children() {
        collect_memory_accesses(expr, usize::from(child_id), reads, writes);
    }
}

#[derive(Debug, Clone)]
pub struct StoreOperation {
    pub base: Id,
    pub val: Id,
    pub index: Id,
    pub position: usize, // position in the sequence
}

#[derive(Debug, Clone)]
pub struct LoadOperation {
    pub base: Id,
    pub index: Id,
    pub node_id: Id, // the actual Load node ID in RecExpr
}

pub fn value_forward_substitution(expr: &mut RecExpr<TileLang>) -> bool {
    let mut modified = false;

    let loop_nodes = find_loop_nodes(expr);

    for loop_id in loop_nodes {
        if let TileLang::Loop([_start, _end, _tile, _loop_var, body]) = &expr[loop_id] {
            let body_id = *body;
            let store_ops = extract_store_sequence(expr, body_id);

            for (i, store_op) in store_ops.iter().enumerate() {
                if is_reduction_operation(expr, store_op) {
                    continue;
                }

                let loads = find_substitutable_loads(expr, &store_ops, i);
                if loads.len() == 1 {
                    substitute_loads(expr, &loads, store_op.val);
                    modified = true;
                }
            }
        }
    }
    modified
}

// Helper function to find all Loop nodes
fn find_loop_nodes(expr: &RecExpr<TileLang>) -> Vec<Id> {
    let mut loop_nodes = Vec::new();

    for (id, node) in expr.as_ref().iter().enumerate() {
        if matches!(node, TileLang::Loop(_)) {
            loop_nodes.push(Id::from(id));
        }
    }

    loop_nodes
}

pub fn extract_store_sequence(expr: &RecExpr<TileLang>, body_id: Id) -> Vec<StoreOperation> {
    let mut stores = Vec::new();
    let mut position = 0;

    fn flatten_seq_recursive(
        expr: &RecExpr<TileLang>,
        node_id: Id,
        stores: &mut Vec<StoreOperation>,
        position: &mut usize,
    ) {
        let node = &expr[node_id];
        match node {
            TileLang::Seq([left, right]) => {
                flatten_seq_recursive(expr, *left, stores, position);
                flatten_seq_recursive(expr, *right, stores, position);
            }
            TileLang::Store([base, val, index]) => {
                stores.push(StoreOperation {
                    base: *base,
                    val: *val,
                    index: *index,
                    position: *position,
                });
                *position += 1;
            }
            TileLang::Loop(_) => {
                *position += 1;
            }
            _ => {}
        }
    }

    flatten_seq_recursive(expr, body_id, &mut stores, &mut position);
    stores
}

pub fn is_reduction_operation(expr: &RecExpr<TileLang>, store_op: &StoreOperation) -> bool {
    fn contains_matching_load(
        expr: &RecExpr<TileLang>,
        node_id: Id,
        target_base: Id,
        target_index: Id,
    ) -> bool {
        let node = &expr[node_id];

        match node {
            TileLang::Load([base, index]) => {
                if bases_equivalent(expr, *base, target_base)
                    && indices_equivalent(expr, *index, target_index)
                {
                    return true;
                }
            }
            _ => {
                for &child_id in node.children() {
                    if contains_matching_load(expr, child_id, target_base, target_index) {
                        return true;
                    }
                }
            }
        }
        false
    }
    contains_matching_load(expr, store_op.val, store_op.base, store_op.index)
}

pub fn find_substitutable_loads(
    expr: &RecExpr<TileLang>,
    store_ops: &[StoreOperation],
    target_position: usize,
) -> Vec<LoadOperation> {
    let mut loads = Vec::new();
    let target_store = &store_ops[target_position];

    for store_op in store_ops.iter().skip(target_position + 1) {
        find_loads_in_subtree(
            expr,
            store_op.val,
            target_store.base,
            target_store.index,
            &mut loads,
        );
    }
    loads
}

fn find_loads_in_subtree(
    expr: &RecExpr<TileLang>,
    node_id: Id,
    target_base: Id,
    target_index: Id,
    loads: &mut Vec<LoadOperation>,
) {
    let node = &expr[node_id];

    match node {
        TileLang::Load([base, index]) => {
            if bases_equivalent(expr, *base, target_base)
                && indices_equivalent(expr, *index, target_index)
            {
                loads.push(LoadOperation {
                    base: *base,
                    index: *index,
                    node_id,
                });
            }
        }
        _ => {
            for &child_id in node.children() {
                find_loads_in_subtree(expr, child_id, target_base, target_index, loads);
            }
        }
    }
}

pub fn substitute_loads(
    expr: &mut RecExpr<TileLang>,
    loads: &[LoadOperation],
    substitute_value: Id,
) {
    for load_op in loads {
        expr[load_op.node_id] = expr[substitute_value].clone();
    }
}

// Enhanced base comparison using concrete names
pub fn bases_equivalent(expr: &RecExpr<TileLang>, base1: Id, base2: Id) -> bool {
    let base1_name = get_base_name(expr, usize::from(base1));
    let base2_name = get_base_name(expr, usize::from(base2));

    match (base1_name, base2_name) {
        (Some(name1), Some(name2)) => name1 == name2,
        _ => false,
    }
}

// Enhanced index comparison using concrete nodes
pub fn indices_equivalent(expr: &RecExpr<TileLang>, index1: Id, index2: Id) -> bool {
    let node1 = get_index_node(expr, usize::from(index1));
    let node2 = get_index_node(expr, usize::from(index2));

    match (node1, node2) {
        (Some(n1), Some(n2)) => nodes_equivalent(expr, &n1, &n2),
        _ => false,
    }
}

// Recursive node equivalence checker
fn nodes_equivalent(expr: &RecExpr<TileLang>, node1: &TileLang, node2: &TileLang) -> bool {
    match (node1, node2) {
        (TileLang::Var(v1), TileLang::Var(v2)) => v1 == v2,
        (TileLang::Num(n1), TileLang::Num(n2)) => n1 == n2,
        (TileLang::Add([a1, b1]), TileLang::Add([a2, b2])) => {
            // Check both orders for commutative operations
            (indices_equivalent(expr, *a1, *a2) && indices_equivalent(expr, *b1, *b2))
                || (indices_equivalent(expr, *a1, *b2) && indices_equivalent(expr, *b1, *a2))
        }
        (TileLang::Sub([a1, b1]), TileLang::Sub([a2, b2])) => {
            // Non-commutative
            indices_equivalent(expr, *a1, *a2) && indices_equivalent(expr, *b1, *b2)
        }
        (TileLang::Mul([a1, b1]), TileLang::Mul([a2, b2])) => {
            // Commutative
            (indices_equivalent(expr, *a1, *a2) && indices_equivalent(expr, *b1, *b2))
                || (indices_equivalent(expr, *a1, *b2) && indices_equivalent(expr, *b1, *a2))
        }
        (TileLang::Div([a1, b1]), TileLang::Div([a2, b2])) => {
            // Non-commutative
            indices_equivalent(expr, *a1, *a2) && indices_equivalent(expr, *b1, *b2)
        }
        (TileLang::Tile(t1), TileLang::Tile(t2)) => indices_equivalent(expr, *t1, *t2),
        (TileLang::Elem(e1), TileLang::Elem(e2)) => indices_equivalent(expr, *e1, *e2),
        (TileLang::Index(idx1), TileLang::Index(idx2)) => {
            // Compare index arrays
            if idx1.len() != idx2.len() {
                return false;
            }
            idx1.iter()
                .zip(idx2.iter())
                .all(|(i1, i2)| indices_equivalent(expr, *i1, *i2))
        }
        (TileLang::FullTile, TileLang::FullTile) => true,
        (TileLang::ConstTile([a1, b1]), TileLang::ConstTile([a2, b2])) => {
            // Non-commutative
            indices_equivalent(expr, *a1, *a2) && indices_equivalent(expr, *b1, *b2)
        }
        _ => false,
    }
}
