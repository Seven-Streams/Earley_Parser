from dataclasses import dataclass, field
from typing import List, Set, Tuple, Union, Dict

global_rule_dict: dict[str, int] = {}
@dataclass
class NFA:
    name: str
    init_node: int = 0
    node_cnt: int = 1
    final_node: Set[int] = field(default_factory=set)
    # the key is the node which the transition is from.
    # the value is a union. the first element shows the action,
    # i.e. the corresponding rule. the second element is the node
    # which the transition is to.
    transitions: Dict[int, Tuple[Union[str, int], int]] = field(default_factory=dict)
    def Build(self, input: str):
        lhs, rhs = input.split("::=")
        lhs = lhs.replace(" ", "")
        if(lhs != self.name):
            print(lhs, len(lhs), self.name, len(self.name))
            raise ValueError("The name of the NFA should be the same as the lhs of the rule.")
        for rule in rhs.split("|"):
            current = self.init_node
            for symbol in rule.split(" "):
                if(symbol == ""):
                    continue
                if(self.transitions.get(current) == None):
                    self.transitions[current] = []
                rule_symbol = Union[str, int]
                if(symbol in global_rule_dict):
                    rule_symbol = global_rule_dict[symbol]
                else:
                    rule_symbol = symbol
                flag = False
                for transition in self.transitions[current]:
                    if(transition[0] == rule_symbol):
                        flag = True
                        current = transition[1]
                        break
                if(not flag):
                    self.transitions[current].append((rule_symbol, self.node_cnt))
                    current = self.node_cnt
                    self.node_cnt += 1
            self.final_node.add(current)
test = NFA("test")
test.Build("test ::= a b c")
test.Build("test ::= a b d")
test.Build("test ::= a c e")
print(test)