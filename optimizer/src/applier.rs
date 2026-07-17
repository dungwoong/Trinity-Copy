//! Custom appliers for complex rewrite rules

use crate::dependency::*;
use crate::language::LoopAnalysis;
use crate::language::TileLang;
use crate::utils::*;
use egg::*;

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

pub struct LoopSplit {
    pub end: Var,
    pub tile: Var,
    pub loop_var: Var,
    pub a: Var,
    pub idx: Var,
    pub body: Var,
    pub new_tile: Var,
    pub new_loop_var: Var,
    pub new_a: Var,
    pub others: Var,
    pub rhs: Pattern<TileLang>,
}

impl Applier<TileLang, LoopAnalysis> for LoopSplit {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<TileLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let mut subst = subst.clone();

        // Generate new variable names based on existing ones
        fn rename_var(egraph: &mut EGraph, id: Id, prefix: &str) -> Id {
            for node in &egraph[id].nodes {
                if let TileLang::Var(sym) = node {
                    let new_sym = format!("{}_{}", prefix, sym);
                    return egraph.add(TileLang::Var(new_sym.into()));
                }
            }
            // Fallback if not a variable
            egraph.add(TileLang::Var(
                format!("{}_{}", prefix, id.as_usize()).into(),
            ))
        }

        let new_tile_id = rename_var(egraph, subst[self.tile], "new");
        let new_loop_var_id = rename_var(egraph, subst[self.loop_var], "new");
        let new_a_id = rename_var(egraph, subst[self.a], "new");

        // Insert the new variable bindings into the substitution
        subst.insert(self.new_tile, new_tile_id);
        subst.insert(self.new_loop_var, new_loop_var_id);
        subst.insert(self.new_a, new_a_id);

        // Apply the RHS pattern using the updated substitution
        self.rhs
            .apply_one(egraph, eclass, &subst, searcher_ast, rule_name)
    }
}

pub struct LoopSplitTail {
    pub end: Var,
    pub tile: Var,
    pub loop_var: Var,
    pub a: Var,
    pub idx: Var,
    pub body: Var,
    pub new_tile: Var,
    pub new_loop_var: Var,
    pub new_a: Var,
    pub others: Var,
    pub rhs: Pattern<TileLang>,
}

impl Applier<TileLang, LoopAnalysis> for LoopSplitTail {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<TileLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let mut subst = subst.clone();

        // Generate new variable names based on existing ones
        fn rename_var(egraph: &mut EGraph, id: Id, prefix: &str) -> Id {
            for node in &egraph[id].nodes {
                if let TileLang::Var(sym) = node {
                    let new_sym = format!("{}_{}", prefix, sym);
                    return egraph.add(TileLang::Var(new_sym.into()));
                }
            }
            // Fallback if not a variable
            egraph.add(TileLang::Var(
                format!("{}_{}", prefix, id.as_usize()).into(),
            ))
        }

        let new_tile_id = rename_var(egraph, subst[self.tile], "new");
        let new_loop_var_id = rename_var(egraph, subst[self.loop_var], "new");
        let new_a_id = rename_var(egraph, subst[self.a], "new");

        // Insert the new variable bindings into the substitution
        subst.insert(self.new_tile, new_tile_id);
        subst.insert(self.new_loop_var, new_loop_var_id);
        subst.insert(self.new_a, new_a_id);

        // Apply the RHS pattern using the updated substitution
        self.rhs
            .apply_one(egraph, eclass, &subst, searcher_ast, rule_name)
    }
}

pub struct LoopFusion {
    pub tile1: Var,
    pub tile2: Var,
    pub n: Var,
    pub m: Var,
    pub loop_var1: Var,
    pub loop_var2: Var,
    pub body1: Var,
    pub body2: Var,
    pub start: Var,
}

impl Applier<TileLang, LoopAnalysis> for LoopFusion {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<TileLang>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let start = subst[self.start];
        let n = subst[self.n];
        let m = subst[self.m];
        let tile1 = subst[self.tile1];
        let tile2 = subst[self.tile2];
        let loop_var1 = subst[self.loop_var1];
        let loop_var2 = subst[self.loop_var2];
        let body1 = subst[self.body1];
        let body2 = subst[self.body2];

        let no_dep =
            no_raw_dependency(self.body1, self.body2, self.loop_var1)(egraph, eclass, subst);

        let is_num_tile1 = is_num(self.tile1)(egraph, eclass, subst);
        let is_num_tile2 = is_num(self.tile2)(egraph, eclass, subst);

        let cond1_nm = cond1(self.n, self.tile1, self.m)(egraph, eclass, subst);
        let cond1_mn = cond1(self.m, self.tile2, self.n)(egraph, eclass, subst);

        let mut results = vec![];

        if !no_dep {
            return vec![];
        }

        if loop_var1 == loop_var2 && n == m {
            // same loop var, same range
            if !is_num_tile1 && !is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                results.push(egraph.add(fused));
            } else if is_num_tile1 && !is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                results.push(egraph.add(fused));
            } else if !is_num_tile1 && is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile2, loop_var1, new_body]);
                results.push(egraph.add(fused));
            } else if is_num_tile1 && is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                results.push(egraph.add(fused));
            }
        } else {
            // different loop vars or range
            if is_num_tile1 && !is_num_tile2 && cond1_nm {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::TLoop([start, n, tile1, loop_var1, loop_var2, new_body]);
                results.push(egraph.add(fused));
            } else if !is_num_tile1 && is_num_tile2 && cond1_mn {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::TLoop([start, m, tile2, loop_var2, loop_var1, new_body]);
                results.push(egraph.add(fused));
            }
        }

        for id in &results {
            egraph.union(eclass, *id);
        }

        results
    }
}

pub struct LoopFusionTail {
    pub start: Var,
    pub n: Var,
    pub m: Var,
    pub tile1: Var,
    pub tile2: Var,
    pub loop_var1: Var,
    pub loop_var2: Var,
    pub body1: Var,
    pub body2: Var,
    pub others: Var,
}
impl Applier<TileLang, LoopAnalysis> for LoopFusionTail {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<TileLang>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let start = subst[self.start];
        let n = subst[self.n];
        let m = subst[self.m];
        let tile1 = subst[self.tile1];
        let tile2 = subst[self.tile2];
        let loop_var1 = subst[self.loop_var1];
        let loop_var2 = subst[self.loop_var2];
        let body1 = subst[self.body1];
        let body2 = subst[self.body2];
        let others = subst[self.others];

        let no_dep =
            no_raw_dependency(self.body1, self.body2, self.loop_var1)(egraph, eclass, subst);
        let is_num_tile1 = is_num(self.tile1)(egraph, eclass, subst);
        let is_num_tile2 = is_num(self.tile2)(egraph, eclass, subst);
        let cond1_nm = cond1(self.n, self.tile1, self.m)(egraph, eclass, subst);
        let cond1_mn = cond1(self.m, self.tile2, self.n)(egraph, eclass, subst);

        let mut results = vec![];

        if !no_dep {
            return vec![];
        }

        if loop_var1 == loop_var2 && n == m {
            // same loop var, same range
            if !is_num_tile1 && !is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            } else if is_num_tile1 && !is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            } else if !is_num_tile1 && is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile2, loop_var1, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            } else if is_num_tile1 && is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            }
        } else {
            // different loop vars or range
            if is_num_tile1 && !is_num_tile2 && cond1_nm {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::TLoop([start, n, tile1, loop_var1, loop_var2, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            } else if !is_num_tile1 && is_num_tile2 && cond1_mn {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::TLoop([start, m, tile2, loop_var2, loop_var1, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            }
        }

        for id in &results {
            egraph.union(eclass, *id);
        }

        results
    }
}
