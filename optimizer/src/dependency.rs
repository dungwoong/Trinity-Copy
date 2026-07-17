//! Dependency analysis predicates for rewrite rules

use crate::language::LoopAnalysis;
use crate::language::TileLang;
use crate::utils::*;
use egg::{Id, Subst, Var};

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

// body1과 body2가 loopvar에 대해서 raw hazard가 없음
pub fn no_raw_dependency(
    body1_var: Var,
    body2_var: Var,
    loop_var: Var,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body1_id = subst[body1_var];
        let body2_id = subst[body2_var];

        let loop_var_str = egraph[subst[loop_var]]
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

        let (d1_reads, d1_writes) = collect_access_sets(egraph, body1_id, false);
        let (d2_reads, _d2_writes) = collect_access_sets(egraph, body2_id, false);
        // println!("{:?}", d2_reads);
        // println!("{:?}", d1_writes);
        // println!("=================");

        for r2 in &d2_reads {
            for w1 in &d1_writes {
                if base_overlap(r2, w1) {
                    // New logic: if bases overlap but indices are different, return false
                    if !indices_are_same(egraph, &r2.index, &w1.index) {
                        // println!("Write: {:?}", w1);
                        // println!("Read: {:?}", r2);
                        return false;
                    }

                    if has_cross_iteration_dependency(w1, &d1_reads, &loop_var_str, egraph) {
                        // println!("Dependency!");
                        // println!("{:?}", w1);
                        // println!("{:?}", loop_var_str);
                        return false;
                    }
                }
            }
        }
        true
    }
}

// body1과 body2가 전혀 관계 없음
pub fn no_all_dependency(
    body1_var: Var,
    body2_var: Var,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body1_id = subst[body1_var];
        let body2_id = subst[body2_var];

        let (d1_reads, d1_writes) = collect_access_sets(egraph, body1_id, false);
        let (d2_reads, d2_writes) = collect_access_sets(egraph, body2_id, false);

        // println!("d1_reads: {:?}\n d1_writes: {:?}", d1_reads, d1_writes);
        // println!("d2_reads: {:?}\n d2_writes: {:?}", d2_reads, d2_writes);

        for r2 in &d2_reads {
            for w1 in &d1_writes {
                if base_overlap(r2, w1) {
                    return false;
                }
            }
        }
        for w2 in &d2_writes {
            for r1 in &d1_reads {
                if base_overlap(w2, r1) {
                    return false;
                }
            }
        }
        for w2 in &d2_writes {
            for w1 in &d1_writes {
                if base_overlap(w2, w1) {
                    return false;
                }
            }
        }

        true
    }
}

pub fn no_dependency_with_base(
    body_var: Var,
    base_var: Var,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body_id = subst[body_var];
        let base_id = subst[base_var];

        let (reads, writes) = collect_access_sets(egraph, body_id, false);
        let base = get_base_name_egraph(egraph, base_id);

        for r in &reads {
            if r.base.as_ref().map_or(false, |b| {
                base.as_ref()
                    .map_or(false, |base_str| bases_overlap(b, base_str))
            }) {
                return false;
            }
        }
        for w in &writes {
            if w.base.as_ref().map_or(false, |b| {
                base.as_ref()
                    .map_or(false, |base_str| bases_overlap(b, base_str))
            }) {
                return false;
            }
        }
        true
    }
}

// body의 read/write set이 loop_var의 영향을 전혀 받지 않음
pub fn no_dependency_with_loopvar(
    body: Var,
    loop_var: Var,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body_id = subst[body];
        let loop_var_id = subst[loop_var];

        // extract the variable name string (e.g., "n")
        let loop_var_str = egraph[loop_var_id]
            .nodes
            .iter()
            .find_map(|n| {
                if let TileLang::Var(sym) = n {
                    Some(sym.as_str().to_string())
                } else {
                    None
                }
            })
            .expect("loop_var must be a Var");

        let (read_set, write_set) = collect_access_sets(egraph, body_id, false);

        for access in read_set.iter().chain(write_set.iter()) {
            for index in &access.index {
                if index_depends_on(index, egraph, &loop_var_str) {
                    return false;
                }
            }
            // if let Some(index) = &access.index {
            //     if index_depends_on(index, egraph, &loop_var_str) {
            //         return false;
            //     }
            // }
        }

        true
    }
}

pub fn has_dependency_with_loopvar(
    body: Var,
    loop_var: Var,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body_id = subst[body];
        let loop_var_id = subst[loop_var];

        // extract the variable name string (e.g., "n")
        let loop_var_str = egraph[loop_var_id]
            .nodes
            .iter()
            .find_map(|n| {
                if let TileLang::Var(sym) = n {
                    Some(sym.as_str().to_string())
                } else {
                    None
                }
            })
            .expect("loop_var must be a Var");

        let (read_set, write_set) = collect_access_sets(egraph, body_id, false);

        for access in read_set.iter().chain(write_set.iter()) {
            for index in &access.index {
                if index_depends_on(index, egraph, &loop_var_str) {
                    return true;
                }
            }
            // if let Some(index) = &access.index {
            //     if index_depends_on(index, egraph, &loop_var_str) {
            //         return true;
            //     }
            // }
        }

        false
    }
}
