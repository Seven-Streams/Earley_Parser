from dataclasses import dataclass, field
from typing import List, Set, Tuple, Union, Dict

global_rule_dict: dict[str, int] = {}
ROOT_RULE= "$"
root_rule_number = 0
EPSILON = "EPSILON"
# The following flags are used to represent a universal symbol.
XGRAMMAR_EVERYTHING_FLAG = "EVERYTHING"
XGRAMMAR_HEX_FLAG = "HEX"
XGRAMMAR_DIGIT_FLAG = "DIGIT"
LOOP_FLAG = "LOOP_FLAG"
DEF_FLAG = "DEF_FLAG"
FORCE_FLAG = "FORCE_FLAG"
NEED_LOOP_FLAG = "NEED_LOOP_FLAG"
NEED_DEF_FLAG = "NEED_DEF_FLAG"
IF_FLAG = "IF_FLAG"
NEED_IF_FLAG = "NEED_IF_FLAG"
COMPLETE_FLAG = "COMPLETE_FLAG"
WHITE_SPACE_FLAG = "WHITE_SPACE_FLAG"
OR_FLAG = "OR_FLAG"
VARIABLE_FLAG = "VARIABLE_FLAG"

loop_rules = set()
def_rules = set()
force_rules = set()
if_rules = set()
need_loop_rules = set()
need_def_rules = set()
need_if_rules = set()
complete_line_rules = set()
global_rule_dict = {}

def is_terminal(symbol: Union[str, int]) -> bool:
    return isinstance(symbol, str)

@dataclass
class NFA:
    name: int
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
        if(global_rule_dict[lhs] != self.name):
            print(global_rule_dict[lhs], self.name)
            raise ValueError("The name of the NFA should be the same as the lhs of the rule.")
        for rule in rhs.split("|"):
            current = self.init_node
            for symbol in rule.split(" "):
                if(symbol == ""):
                    continue
                if(self.CheckFlags(symbol)):
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

    def Accepted(self, node: int) -> bool:
        return node in self.final_node
            
    def GetTransitions(self, node: int) -> List[Tuple[Union[str, int], int]]:
        if(node not in self.transitions):
            return []
        return self.transitions[node] 
    
    # Check the flags of the rule.
    def CheckFlags(self, symbol:str) -> bool:
        if(symbol == LOOP_FLAG):
            loop_rules.add(self.name)
            return True
        if(symbol == DEF_FLAG):
            def_rules.add(self.name)
            return True
        if(symbol == FORCE_FLAG):
            force_rules.add(self.name)
            return True
        if(symbol == NEED_LOOP_FLAG):
            need_loop_rules.add(self.name)
            return True
        if(symbol == NEED_DEF_FLAG):
            need_def_rules.add(self.name)
            return True
        if(symbol == IF_FLAG):
            if_rules.add(self.name)
            return True
        if(symbol == NEED_IF_FLAG):
            need_if_rules.add(self.name)
            return True
        if(symbol == COMPLETE_FLAG):
            complete_line_rules.add(self.name)
            return True    
        return False
@dataclass(frozen=True)
class Grammar:
    NFAs: Dict[int, NFA] = field(default_factory=dict)
    def parse(rule: str) -> "Grammar":
        global global_rule_dict
        NFAs_tmp: Dict[int, NFA] = {}
        cnt = 0
        results = []
        # Do the first scanning. Add all the non-terminal symbols to the dictionary.
        for line in rule.split("\n"):
            if not line.strip():
                continue
            lhs, rhs = line.replace(" ", "").split("::=")
            cnt += 1
            if (lhs not in global_rule_dict):
                if lhs == ROOT_RULE:
                    global_rule_dict[lhs] = root_rule_number
                else:
                    global_rule_dict[lhs] = cnt
            if(global_rule_dict[lhs] not in NFAs_tmp):
                NFAs_tmp[global_rule_dict[lhs]] = NFA(global_rule_dict[lhs])
        # Do the second scanning. Replace the non-terminal symbols with the corresponding number.
        cnt = 0
        for line in rule.split("\n"):
            if not line.strip():
                continue
            lhs, _ = line.replace(" ", "").split("::=")
            lhs = lhs.replace(" ", "")
            NFAs_tmp[global_rule_dict[lhs]].Build(line)
        return Grammar(NFAs_tmp)

    def __getitem__(self, symbol: str) -> NFA:
        return self.NFAs[symbol]

@dataclass(frozen=True)
class State:
    rule_name: int
    node_num: int
    pos: int
    accepted: bool
    def terminated(self) -> bool:
        return self.accepted
    def __repr__(self):
        return f"State({self.rule_name}, {self.node_num}, {self.pos}, {self.accepted})"
    def __hash__(self):
        return hash((self.rule_name, self.node_num, self.pos, self.accepted))

@dataclass
class Parser:
    grammar: Grammar
    loop_indent: List[int]
    if_indent: List[int]
    define: bool = False
    force_indent:bool = False
    now_indent: int = 0
    white_space_cnt: int = 0
    state_num = 0
    tokens_num = 0
    tokens_num_without_indent = 0
    lines = 0
    complete_times = 0
    scan_times = 0
    predict_times = 0
    init_times = 0
    line_start: bool = True
    def __post_init__(self):
        self.states: List[Dict[int, Set[State]]] = []
        self.input = ""
        self.next_states: Set[State] = set()
        self.current_states: Set[State] = set()
        self.current_states.add(State(root_rule_number, 0, 0, self.GetAccepted(root_rule_number, 0)))
    
    def _complete(self, state: State):
        if(state.rule_name in self.states[state.pos]):
            for parent_state in self.states[state.pos][state.rule_name]:
                transitions = self.grammar.NFAs[parent_state.rule_name].GetTransitions(parent_state.node_num)
                for trans in transitions:
                    if trans[0] == state.rule_name:
                        self.queue.append(State(parent_state.rule_name, trans[1], parent_state.pos, self.GetAccepted(parent_state.rule_name, trans[1])))    
    
    def _scan_predict(self, state: State, token: str):
        transitions = self.grammar.NFAs[state.rule_name].GetTransitions(state.node_num)
        for trans in transitions:
            # Scanning.
            if is_terminal(trans[0]):
                if trans[0] == EPSILON:
                    self.queue.append(State(state.rule_name, trans[1], state.pos, self.GetAccepted(state.rule_name, trans[1])))
                if ((trans[0] == token) 
                    or (trans[0] == XGRAMMAR_EVERYTHING_FLAG and token != "\\")
                    or (trans[0] == XGRAMMAR_DIGIT_FLAG and token.isdigit())
                    or (trans[0] == XGRAMMAR_HEX_FLAG and token in "0123456789abcdefABCDEF")
                    or (trans[0] == WHITE_SPACE_FLAG and token == " ")
                    or (trans[0] == VARIABLE_FLAG and token in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
                    or (trans[0] ==OR_FLAG and token == "|")):
                    self.next_states.add(State(state.rule_name, trans[1], state.pos, self.GetAccepted(state.rule_name, trans[1])))
            else:
            # Predicting.
                if(int(trans[0]) not in self.states[len(self.states) - 1]):
                    self.states[len(self.states) - 1][int(trans[0])] = set()
                self.states[len(self.states) - 1][int(trans[0])].add(state)
                self.queue.append(State(trans[0], 0, len(self.states) - 1, self.GetAccepted(state.rule_name, 0)))
            
    
    def _consume(self, token: str):
        self.states.append(dict())
        self.input += token
        self.queue = [s for s in self.current_states]
        self.current_states.clear()
        while self.queue:
            state = self.queue.pop(0)
            if state in self.current_states:
                continue
            self.current_states.add(state)
            if state.accepted:
                self._complete(state)
            self._scan_predict(state, token)
        self.current_states = self.next_states
        self.next_states = set()
    
    def read(self, text: str):
        for token in text:
            indent = self.CheckIndent(token)
            if indent:
                continue
            if token == "\n":
                self.lines += 1
                self.MarkIndent()
                continue
            self._consume(token)
        return self
        
        
    def MarkIndent(self):
        self.states.append(dict())
        self.queue = [s for s in self.current_states]
        self.current_states.clear()
        while self.queue:
            state = self.queue.pop(0)
            if state in self.current_states:
                continue
            self.current_states.add(state)
            if state.accepted:
                self._complete(state)
        for s in self.current_states:
            if s.accepted and complete_line_rules.__contains__(s.rule_name):
                #Detect the loop.
                if force_rules.__contains__(s.rule_name):
                    self.force_indent = True
                else:
                    self.force_indent = False
                if loop_rules.__contains__(s.rule_name):
                    self.loop_indent.append(self.now_indent)
                if if_rules.__contains__(s.rule_name):
                    self.if_indent.append(self.now_indent)
                #To check the new lines.
                if def_rules.__contains__(s.rule_name):
                    self.define = True
                if need_loop_rules.__contains__(s.rule_name):
                    if self.loop_indent == []:
                        raise Exception("Indentation Error: Need Loop")
                if need_def_rules.__contains__(s.rule_name):
                    if not self.define:
                        raise Exception("Indentation Error: Need Define")
                if need_if_rules.__contains__(s.rule_name):
                    if not self.now_indent in self.if_indent:
                        raise Exception("Indentation Error: If else Error")
                self.__post_init__()
                return
        raise Exception("Syntax Error!")    
        
    def GetAccepted(self, rule:int, node:int) -> bool:
        return self.grammar.NFAs[rule].Accepted(node)
    
    def Accepted(self) -> bool:
        for state in self.current_states:
            if state.rule_name == root_rule_number and state.accepted:
                return True
        return False
    
    def CheckIndent(self, token: str) -> bool:
        if token == " " and self.line_start:
            self.tokens_num += 1
            self.white_space_cnt += 1 
            return True
        
        if token == "\n":
            # An empty line.
            if(self.line_start):
                return True
            self.line_start = True
            self.white_space_cnt = 0
            
        if token != " " and token != "\n" and self.line_start:
            # Indentation should be multiple of 4.
            if self.white_space_cnt % 4 != 0:
                raise Exception("Indentation Error: Not Multiple of 4")
            # Too deep indentation.
            if (self.now_indent < self.white_space_cnt - 4):
                raise Exception("Indentation Error: Too Deep Indentation")
            if (self.now_indent == self.white_space_cnt - 4) and not self.force_indent:
                raise Exception("Indentation Error: Too Deep Indentation")
            # The forced indentation is not satisfied.
            if self.force_indent and self.now_indent != self.white_space_cnt - 4:
                raise Exception("Indentation Error: Forced Indentation")
            # The same indentation.
            if self.now_indent == self.white_space_cnt:
                pass
            # The indentation is appended.
            if self.now_indent == self.white_space_cnt - 4:
                self.now_indent = self.white_space_cnt
            # The indentation is popped.
            if self.now_indent > self.white_space_cnt:
                self.now_indent = self.white_space_cnt
                if(self.now_indent == 0):
                    self.define = False
                    while(self.loop_indent != [] and self.loop_indent[-1] > self.now_indent):
                        self.loop_indent.pop()
                    while(self.if_indent != [] and self.if_indent[-1] > self.now_indent):
                        self.if_indent.pop()
            self.line_start = False
        return False

    def _print(self):
        assert(len(self.current_states) > 0)
        # print(self.input ,len(self.current_states) > 0)

grammar = Grammar.parse(
    """
    $ ::= python
    python ::= python whitespaces | if_statement | else_statement | while_statement | for_statement | break_statement | continue_statement | return_statement | function_definition | expr_statement
    function_definition ::= d e f whitespaces variable ( args ) : FORCE_FLAG DEF_FLAG COMPLETE_FLAG | d e f whitespaces variable ( ) : FORCE_FLAG DEF_FLAG COMPLETE_FLAG 
    args ::= args , variable | variable
    Float ::= Int . Int 
    Int ::= DIGIT | Int DIGIT 
    Array ::= expr [ expr ] | expr [ ] | { } | [ ] | [ in_args ]
    Dict ::= { } | { key_values }
    key_values ::= key_values , expr : expr | expr : expr
    String ::= " " | " chars " | ' ' | ' chars '
    expr_statement ::= expr COMPLETE_FLAG
    if_statement ::= i f whitespaces expr : FORCE_FLAG IF_FLAG COMPLETE_FLAG
    else_statement ::= e l s e : NEED_IF_FLAG COMPLETE_FLAG FORCE_FLAG
    while_statement ::= w h i l e whitespaces expr : LOOP_FLAG COMPLETE_FLAG FORCE_FLAG
    for_statement ::= f o r whitespaces variable whitespaces i n whitespaces expr : LOOP_FLAG COMPLETE_FLAG FORCE_FLAG
    break_statement ::= b r e a k NEED_LOOP_FLAG COMPLETE_FLAG
    continue_statement ::= c o n t i n u e NEED_LOOP_FLAG COMPLETE_FLAG
    return_statement ::= r e t u r n whitespaces expr NEED_DEF_FLAG COMPLETE_FLAG | r e t u r n NEED_DEF_FLAG COMPLETE_FLAG
    assign_op ::= = | + = | - = | * = | / = | % = | & =  | ^ = | < < = | > > = | * * = | OR_FLAG = | / / =
    expr_binary_op ::= + | - | * | / | % | & | ^ | < < | > > | * * | = = | ! = | < | > | < = | > = | a n d | o r | OR_FLAG | / / | i s | i n
    expr_unary_op ::= + | - | ~ | n o t
    in_args ::= in_args , expr | expr
    func_call ::= variable ( in_args ) | variable ( )
    expr ::= expr_raw | whitespaces expr_raw | expr whitespaces | whitespaces expr whitespaces
    expr_raw ::= Int | Float | String | Bool | variable | func_call | expr expr_binary_op expr | expr_unary_op expr | ( expr ) | expr assign_op expr | Array | expr . expr | expr , expr | Dict
    variable ::= variable_raw | whitespaces variable_raw | variable whitespaces | whitespaces variable whitespaces
    variable_raw ::= variable_char | variable_raw variable_char | variable_raw DIGIT
    variable_char ::= VARIABLE_FLAG
    chars ::= EVERYTHING | chars EVERYTHING | chars escaped | escaped
    escaped ::= \\ \\ | \\ "  | \\ n  | \\ b  | \\ f  | \\ r | \\ t 
    Bool ::= T r u e | F a l s e
    whitespaces ::= WHITE_SPACE_FLAG | whitespaces WHITE_SPACE_FLAG
    """
)
# for nfa in grammar.NFAs:
#     print(nfa)
Parser(grammar, [], []).read(
    """
def count_even_numbers(limit):
    count = 0
    number += 1
    while number <= limit:
        if number % 2 == 0:
            count = count + 1
        number = number + 1
    return count

limit = 10
result = count_even_numbers(limit)

if result > 0:
    print("There are",result, "even numbers.")
else:
    print("No even numbers found.")
    """
)