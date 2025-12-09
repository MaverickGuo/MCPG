import z3
import sys
import time
import os
import argparse

# --- 核心工具函数 ---

def fast_is_int_constraint(expr):
    """
    极速过滤 (O(1)): 检查是否包含 Int 类型运算。
    专门用来秒删那个巨大的 sum/metalLength 语句。
    """
    if z3.is_int(expr):
        return True
    if expr.num_args() > 0:
        # 检查第一个子节点，足以判断 (= metalLength ...) 这种结构
        if z3.is_int(expr.arg(0)):
            return True
    return False

def save_cnf_and_map(goal, cnf_path, map_path):
    """保存 CNF 和 Map，返回 (变量数, 子句数)"""
    var_map = {}
    next_id = 1
    clauses = []
    
    # 遍历 Goal 中的所有子句
    for clause_expr in goal:
        lits = []
        if z3.is_or(clause_expr):
            lits = clause_expr.children()
        else:
            lits = [clause_expr]
            
        dimacs_clause = []
        for lit in lits:
            sign = 1
            atom = lit
            if z3.is_not(lit):
                atom = lit.children()[0]
                sign = -1
            
            if atom not in var_map:
                var_map[atom] = next_id
                next_id += 1
            dimacs_clause.append(sign * var_map[atom])
        clauses.append(dimacs_clause)

    # 1. 写入 CNF
    with open(cnf_path, 'w') as f:
        f.write(f"p cnf {len(var_map)} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")

    # 2. 写入 Map
    with open(map_path, 'w') as f:
        f.write("ID | Variable\n")
        sorted_map = sorted(var_map.items(), key=lambda x: x[1])
        for atom, vid in sorted_map:
            name = str(atom).replace('\n', ' ')
            f.write(f"{vid} | {name}\n")
            
    return len(var_map), len(clauses)

# --- 主流程 ---

def run_pipeline(smt2_file, max_effort):
    print(f"🚀 开始处理: {smt2_file}")
    print(f"⚙️  最大 Effort 设置: {max_effort} (0=Raw, 1=Lite, 2=Deep)")
    print("-" * 60)
    
    t_start = time.time()
    
    # [Step 1] 解析与过滤 (预处理)
    print("Step 1: 解析并过滤 Int 约束...", end="", flush=True)
    try:
        assertions = z3.parse_smt2_file(smt2_file)
    except Exception as e:
        print(f"\n❌ 解析失败: {e}")
        return

    g = z3.Goal()
    dropped_count = 0
    for a in assertions:
        if fast_is_int_constraint(a):
            dropped_count += 1
        else:
            g.add(a)
    
    t_parse = time.time()
    print(f" ✅ 完成 ({t_parse - t_start:.2f}s)")
    print(f"        丢弃了 {dropped_count} 条 Int/求和语句")

    stats = []

    # --- Phase 1: Raw Conversion (Effort 0) ---
    # 这是必须跑的，最基础的转换
    print(f"\nStep 2: [Effort 0] 生成 RAW CNF (仅转换)...", end="", flush=True)
    t0 = time.time()
    
    strat_raw = z3.Then('simplify', 'bit-blast', 'tseitin-cnf')
    g_raw = strat_raw(g)[0] 
    
    raw_cnf = smt2_file + ".raw.cnf"
    raw_map = smt2_file + ".raw.map"
    v1, c1 = save_cnf_and_map(g_raw, raw_cnf, raw_map)
    
    t1 = time.time()
    print(f" ✅ 完成 ({t1 - t0:.2f}s)")
    print(f"        输出: {os.path.basename(raw_cnf)}")
    print(f"        规模: {v1} 变量, {c1} 子句")
    stats.append(("RAW (Effort 0)", v1, c1, t1-t0))

    # 如果 Effort 为 0，到这里就结束
    if max_effort < 1:
        print("\n🏁 Effort限制为0，处理结束。")
        return

    # --- Phase 2: Lite BCP (Effort 1) ---
    # 仅使用代数化简和传播，不调用 Solver，安全
    print(f"\nStep 3: [Effort 1] 执行 Lite BCP (代数化简)...", end="", flush=True)
    
    g_input_bcp = z3.Goal()
    for f in g_raw: g_input_bcp.add(f) # 修复了之前的 add bug
    
    # 注意：移除了 'ctx-solver-simplify'，防止卡死
    strat_bcp = z3.Then('propagate-values', 'simplify')
    g_bcp = strat_bcp(g_input_bcp)[0]
    
    bcp_cnf = smt2_file + ".bcp.cnf"
    bcp_map = smt2_file + ".bcp.map"
    v2, c2 = save_cnf_and_map(g_bcp, bcp_cnf, bcp_map)
    
    t2 = time.time()
    print(f" ✅ 完成 ({t2 - t1:.2f}s)")
    print(f"        输出: {os.path.basename(bcp_cnf)}")
    stats.append(("Lite (Effort 1)", v2, c2, t2-t1))

    # 如果 Effort 为 1，到这里就结束
    if max_effort < 2:
        print("\n🏁 Effort限制为1，处理结束。")
        print_summary(stats)
        return

    # --- Phase 3: Deep Optimization (Effort 2) ---
    # 调用 Solver 进行包含消除，极慢，慎用
    print(f"\nStep 4: [Effort 2] 执行 Deep Opt (包含消除 - 极慢)...", end="", flush=True)
    
    g_input_opt = z3.Goal()
    for f in g_bcp: g_input_opt.add(f)
    
    # 修复了之前的 z3.Then bug，改用 Tactic
    strat_opt = z3.Tactic('unit-subsume-simplify')
    
    try:
        g_opt = strat_opt(g_input_opt)[0]
        opt_cnf = smt2_file + ".opt.cnf"
        opt_map = smt2_file + ".opt.map"
        v3, c3 = save_cnf_and_map(g_opt, opt_cnf, opt_map)
        t3 = time.time()
        print(f" ✅ 完成 ({t3 - t2:.2f}s)")
        print(f"        输出: {os.path.basename(opt_cnf)}")
        stats.append(("Deep (Effort 2)", v3, c3, t3-t2))
    except Exception as e:
        print(f"\n⚠️ Deep Opt 失败或被中断: {e}")

    print_summary(stats)

def print_summary(stats):
    print("\n" + "="*70)
    print(f"{'阶段':<15} | {'耗时 (s)':<10} | {'变量数':<10} | {'子句数':<10} | {'压缩率':<10}")
    print("-" * 70)
    
    base_c = stats[0][2]
    for label, v, c, t in stats:
        reduction = 0.0
        if base_c > 0:
            reduction = (1 - c / base_c) * 100
        print(f"{label:<15} | {t:<10.2f} | {v:<10} | {c:<10} | {reduction:.2f}%")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMT2 to CNF Converter (Final)")
    parser.add_argument("file", help="Input SMT2 file")
    # 默认 effort 设置为 0 (Raw)，符合你的最佳实践
    parser.add_argument("--effort", type=int, default=0, choices=[0, 1, 2],
                        help="0=Raw(Fastest, Default), 1=Lite(Safe Opt), 2=Deep(Very Slow)")
    
    args = parser.parse_args()
    run_pipeline(args.file, args.effort)