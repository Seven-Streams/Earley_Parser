from dataclasses import dataclass, field
from typing import List, Set, Tuple, Union, Dict

global_rule_dict: dict[str, int] = {}
ROOT_RULE= "$"
root_rule_number = 0
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
        # lhs is the name of the rule, i.e. the name of the NFA.
        # rhs are the exact rules.
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
                    # The symbol is a non-terminal symbol.
                    rule_symbol = global_rule_dict[symbol]
                else:
                    # The symbol is a terminal symbol.
                    rule_symbol = symbol
                flag = False
                for transition in self.transitions[current]:
                    # The transition is already in the NFA.
                    if(transition[0] == rule_symbol):
                        flag = True
                        current = transition[1]
                        break
                if(not flag):
                    # It's a brand new transition.
                    self.transitions[current].append((rule_symbol, self.node_cnt))
                    current = self.node_cnt
                    self.node_cnt += 1
            self.final_node.add(current)
            
@dataclass(frozen=True)
class Grammar:
    NFAs: Dict[str, NFA] = field(default_factory=dict)
    def parse(rule: str) -> "Grammar":
        global global_rule_dict
        NFAs_tmp: Dict[str, NFA] = {}
        cnt = 0
        results = []
        # Do the first scanning. Add all the non-terminal symbols to the dictionary.
        for line in rule.split("\n"):
            if not line.strip():
                continue
            lhs, rhs = line.replace(" ", "").split("::=")
            cnt += 1
            if (lhs not in global_rule_dict):
                global_rule_dict[lhs] = cnt
            if(lhs not in NFAs_tmp):
                NFAs_tmp[lhs] = NFA(lhs)
        # Do the second scanning. Replace the non-terminal symbols with the corresponding number.
        cnt = 0
        for line in rule.split("\n"):
            if not line.strip():
                continue
            lhs, _ = line.replace(" ", "").split("::=")
            lhs = lhs.replace(" ", "")
            NFAs_tmp[lhs].Build(line)
        return Grammar(NFAs_tmp)

    def __getitem__(self, symbol: str) -> NFA:
        return self.NFAs[symbol]

@dataclass(frozen=True)
class State:
    rule_name: str
    node_num: int
    pos: int
    accepted: bool = False
    def terminated(self) -> bool:
        return self.accepted
    def __repr__(self):
        return f"State({self.rule_name}, {self.node_num}, {self.pos}, {self.accepted})"
    def __hash__(self):
        return hash((self.rule_name, self.node_num, self.pos, self.accepted))


test = NFA("test")
test.Build("test ::= a b c")
test.Build("test ::= a b d")
test.Build("test ::= a c e")
print(test)