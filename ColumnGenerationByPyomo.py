#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

'''
这个对照的例子在 https://mp.weixin.qq.com/s?__biz=MzU0NzgyMjgwNg==&mid=2247486623&idx=1&sn=764424e3443b89fb512e00f51e7823d4&scene=19#wechat_redirect
'''
import pyomo.environ
from pyomo.environ import *
from pyomo.core import *

demand = {3: 25,
          6: 20,
          7: 18}  # width: number
# master roll size
W = 16
cst_name = ['cst' + str(i) for i in range(len(demand))]
pattern_name = ['pattern' + str(i) for i in range(len(demand))]


def create_base_cutting_stock(demand, W):

    initial_patterns = [[0, 0, 0] for _ in range(len(demand))]

    # cutting stock base problem
    rmp = ConcreteModel()

    rmp.pattern = Var(pattern_name, domain=pyomo.environ.NonNegativeReals)

    for i, width in enumerate(demand):
        k = int(W // width)
        initial_patterns[i][i]= k

    # add the demand constraints; supply initial identity columns;
    # filling in as many of a single width on a pattern as possible
    rmp.cst_demand = Constraint(cst_name)
    for i, (width, quantity) in enumerate(demand.items()):
        rmp.cst_demand[cst_name[i]] = \
            sum(initial_patterns[i][j] * rmp.pattern['pattern' + str(j)] for j in range(len(pattern_name))) >= quantity

    def obj_rule_rmp(model):
        return sum(model.pattern[i] for i in pattern_name)
    rmp.obj = Objective(rule=obj_rule_rmp)

    rmp.dual = Suffix(direction=Suffix.IMPORT)

    # ## knapsack cut generator
    sp = ConcreteModel()

    sp.widths = Var(['width' + str(i) for i in range(len(demand))], domain=pyomo.environ.NonNegativeIntegers)

    sp.cst_knapsack = Constraint(
        expr=sum(width * sp.widths['width'+str(i)] for i, (width, quantity) in enumerate(demand.items())) <= W
    )

    return rmp, sp, initial_patterns


def solve_cutting_stock(demand, W, solver, iterations=30):
    rmp, sp, patterns = create_base_cutting_stock(demand, W)

    rmp_solver = SolverFactory('cbc', executable='/opt/homebrew/opt/cbc/bin/cbc')
    sp_solver = SolverFactory('cbc', executable='/opt/homebrew/opt/cbc/bin/cbc')

    for _ in range(iterations):

        rmp_solver.solve(rmp, tee=True)
        rmp.obj.display()
        duals = {cn: rmp.dual[rmp.cst_demand[cn]] for cn in cst_name}

        def obj_rule(model):
            return sum(duals['cst'+str(i)] * sp.widths['width'+str(i)] for i in range(len(demand)))
        sp.obj = Objective(rule=obj_rule, sense=maximize)

        sp_solver.solve(sp, tee=True)
        sp.obj.display()

        if value(sp.obj) <= 1:
            # no better column
            break

        for width, var in sp.widths.items():
            print(width, int(round(value(var))))

        for i, (width, var) in enumerate(sp.widths.items()):
            cut_number = int(round(value(var)))
            patterns[i].append(cut_number)

        rmp.clear()
        pattern_name.append('pattern' + str(len(pattern_name)))

        rmp.pattern = Var(pattern_name, domain=pyomo.environ.NonNegativeReals)
        rmp.cst_demand = Constraint(cst_name)
        for i, (width, quantity) in enumerate(demand.items()):
            rmp.cst_demand[cst_name[i]] = \
                sum(patterns[i][j] * rmp.pattern['pattern' + str(j)] for j in range(len(pattern_name))) >= quantity

        def obj_rule_rmp(model):
            return sum(model.pattern[i] for i in pattern_name)
        rmp.obj = Objective(rule=obj_rule_rmp)

        rmp.dual = Suffix(direction=Suffix.IMPORT)

    del rmp.dual

    return rmp, patterns


if __name__ == '__main__':
    solver = SolverFactory('cbc', executable='/opt/homebrew/opt/cbc/bin/cbc')

    rmp, patterns = solve_cutting_stock(demand, W, solver)

    print('Sheets Required: '+str(int(value(rmp.obj))))
    print('patterns:', patterns)
    for pattern, var in rmp.pattern.items():
        print(pattern, value(var))
