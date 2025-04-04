from dataclasses import dataclass, field
from typing import List, Set, Tuple, Union, Dict
import time

global_rule_dict: dict[str, int] = {}
ROOT_RULE= "$"
root_rule_number = 0
EPSILON = "EPSILON"
# The following flags are used to represent a universal symbol.
XGRAMMAR_EVERYTHING_FLAG = "EVERYTHING"
XGRAMMAR_HEX_FLAG = "HEX"
XGRAMMAR_DIGIT_FLAG = "DIGIT"
FORCE_FLAG = "FORCE_FLAG"
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

def is_universal(symbol: Union[str, int]) -> bool:
    return isinstance(symbol, str) and symbol in {XGRAMMAR_EVERYTHING_FLAG, XGRAMMAR_HEX_FLAG, XGRAMMAR_DIGIT_FLAG, WHITE_SPACE_FLAG, VARIABLE_FLAG, OR_FLAG}
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
    transitions: Dict[int, set[Tuple[Union[str, int], int]]] = field(default_factory=dict)
    # This dict is used to merge the nodes which have the same transition.
    from_node: Dict[int, set[int]] = field(default_factory=dict)
    # This is used to check whether the NFA is easy enough to be inlined.
    easy: bool = True
    
    
    
    def Build(self, input: str):
        # lhs is the name of the rule, i.e. the name of the NFA.
        # rhs are the exact rules.
        self.from_node[0] = set()
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
                plus_signal = False
                star_signal = False
                question_signal = False
                quotation_signal = False
                # Check the regex symbols.
                if(symbol[len(symbol) - 1] == "+"):
                    plus_signal = True
                    symbol = symbol[0:len(symbol) - 1]
                if(symbol[len(symbol) - 1] == "*"):
                    star_signal = True
                    symbol = symbol[0:len(symbol) - 1]
                if(symbol[len(symbol) - 1] == "?"):
                    question_signal = True
                    symbol = symbol[0:len(symbol) - 1]
                if(symbol[0] == '\'' and symbol[len(symbol) - 1] == '\''):
                    quotation_signal = True
                    symbol = symbol[1:len(symbol) - 1]
                if(symbol[0] == '"' and symbol[len(symbol) - 1] == '"'):
                    quotation_signal = True
                    symbol = symbol[1:len(symbol) - 1]              
                if(self.CheckFlags(symbol)):
                    continue
                if(self.transitions.get(current) == None):
                    self.transitions[current] = set()
                rule_symbol_list = []
                if((not quotation_signal) and (not is_universal(symbol))):
                    # The symbol is a non-terminal symbol.
                    if (symbol not in global_rule_dict):
                        raise ValueError("The symbol should be in the global rule dictionary.", symbol)
                    rule_symbol_list.append(global_rule_dict[symbol])
                else:
                    if (is_universal(symbol)):
                        rule_symbol_list.append(symbol)
                    else:
                        escape = False
                        for char in symbol:
                            if char == "\\" and not escape:
                                escape = True
                                continue
                            escape = False
                            rule_symbol_list.append(char)
                            
                    # The symbol is a terminal symbol.
                if plus_signal or star_signal:
                    if not current in self.transitions:
                        self.transitions[current] = set()
                    self.transitions[current].add((EPSILON, self.node_cnt))
                    if(not self.node_cnt in self.from_node):
                        self.from_node[self.node_cnt] = set()
                    self.from_node[self.node_cnt].add(current)
                    current = self.node_cnt
                    self.node_cnt += 1
                start_node = current
                for rule_symbol in rule_symbol_list:
                    flag = False
                    if not is_terminal(rule_symbol):
                        self.easy = False
                    if not current in self.transitions:
                        self.transitions[current] = set()
                    for transition in self.transitions[current]:
                        # The transition is already in the NFA.
                        if(transition[0] == rule_symbol):
                            flag = True
                            current = transition[1]
                            break
                    if(not flag):
                        # It's a brand new transition.
                        self.transitions[current].add((rule_symbol, self.node_cnt))
                        if self.node_cnt not in self.from_node:
                            self.from_node[self.node_cnt] = set()
                        self.from_node[self.node_cnt].add(current)
                        current = self.node_cnt
                        self.node_cnt += 1
                
                if(plus_signal):
                    back_transition = (EPSILON, start_node)
                    if(not current in self.transitions):
                        self.transitions[current] = set()
                    self.transitions[current].add(back_transition)
                    if(not start_node in self.from_node):
                        self.from_node[start_node] = set()
                    self.from_node[start_node].add(current)
    
                if(star_signal):
                    back_transition = (EPSILON, start_node)
                    if(not current in self.transitions):
                        self.transitions[current] = set()
                    self.transitions[current].add(back_transition)
                    if(not start_node in self.from_node):
                        self.from_node[start_node] = set()
                    self.from_node[start_node].add(current)
                    empty_transition = (EPSILON, current)
                    self.transitions[start_node].add(empty_transition)
                    if(not current in self.from_node):
                        self.from_node[current] = set()
                    self.from_node[current].add(start_node)
                
                if(question_signal):
                    empty_transition = (EPSILON, current)
                    if(not start_node in self.transitions):
                        self.transitions[start_node] = set()
                    self.transitions[start_node].add(empty_transition)
                    if(not current in self.from_node):
                        self.from_node[current] = set()
                    self.from_node[current].add(start_node)
                        
            self.final_node.add(current)
        self.Simplify()
        self.ToDFA()


    # this function is used to merge the nodes such like:
    # a --rule--> b
    # c --rule--> b
    # whose outward transitions are the same.
    def Simplify(self):
        flag = True
        deprecated_nodes = set()
        while flag:
            flag = False
            for node in range(0, self.node_cnt):
                if node in deprecated_nodes:
                    continue
                for rhs_node in range(node + 1, self.node_cnt):
                    if rhs_node in deprecated_nodes:
                        continue
                    # The two nodes are equivalent.
                    if (self.transitions.get(node) == self.transitions.get(rhs_node)) and \
                        ((node in self.final_node and rhs_node in self.final_node) or \
                        (node not in self.final_node and rhs_node not in self.final_node)):
                            flag = True
                            deprecated_nodes.add(rhs_node)
                            if rhs_node in self.final_node:
                                self.final_node.remove(rhs_node)
                            for parent_node in self.from_node[rhs_node]:
                                if parent_node in deprecated_nodes:
                                    continue
                                queue = []
                                for trans in self.transitions[parent_node]:
                                    if trans[1] == rhs_node:
                                        queue.append(trans)
                                while queue:
                                    front = queue.pop(0)
                                    self.transitions[parent_node].remove(front)
                                    new_front = (front[0], node)
                                    self.transitions[parent_node].add(new_front)
                                


    def Accepted(self, node: int) -> bool:
        return node in self.final_node
            
    def GetTransitions(self, node: int) -> List[Tuple[Union[str, int], int]]:
        if(node not in self.transitions):
            return []
        return self.transitions[node] 
    
    # Check the flags of the rule.
    def CheckFlags(self, symbol:str) -> bool:
        if(symbol == FORCE_FLAG):
            force_rules.add(self.name)
            return True
        if(symbol == COMPLETE_FLAG):
            complete_line_rules.add(self.name)
            return True    
        return False
    
    # This function is used to convert the NFA to DFA.
    def ToDFA(self) -> None:
        closures: List[Set[int]] = []
        DFA_transitions: Dict[int, set[Tuple[Union[str, int], int]]] = {}
        DFA_final_node: Set[int] = set()
        init_state = self.FindEpsilonClosure(self.init_node)
        closures.append(init_state)
        now_processing = 0
        while now_processing < len(closures):
            clousure_transitions = self.GetClosureTransitions(closures[now_processing])
            for node in closures[now_processing]:
                if node in self.final_node:
                    DFA_final_node.add(now_processing)
            for symbol, nodes in clousure_transitions.items():
                new_closure = set()
                for node in nodes:
                    new_closure = new_closure.union(self.FindEpsilonClosure(node))
                if new_closure not in closures:
                    closures.append(new_closure)
                    closure_num = len(closures) - 1
                else:
                    closure_num = closures.index(new_closure)
                if now_processing not in DFA_transitions:
                    DFA_transitions[now_processing] = set()
                DFA_transitions[now_processing].add((symbol, closure_num))
            now_processing += 1
        self.transitions = DFA_transitions
        self.final_node = DFA_final_node
        self.node_cnt = len(closures)
        self.final_node = DFA_final_node
        self.from_node.clear()
        for from_node, transitions in self.transitions.items():
            for transition in transitions:
                if transition[1] not in self.from_node:
                    self.from_node[transition[1]] = set()
                self.from_node[transition[1]].add(from_node)
        self.init_node = 0
    
    def FindEpsilonClosure(self, node: int) -> Set[int]:
        queue = [node]
        closure = set()
        closure.add(node)
        while queue:
            transitions = self.GetTransitions(queue.pop(0))
            for transition in transitions:
                if transition[0] == EPSILON and transition[1] not in closure:
                    closure.add(transition[1])
                    queue.append(transition[1])
        return closure
    
    def GetClosureTransitions(self, closure: Set[int]) -> Dict[Union[str, int], Set[int]]:
        clousure_transitions: Dict[Union[str, int], Set[int]] = {}
        for node in closure:
            transitions = self.GetTransitions(node)
            for transition in transitions:
                if transition[0] == EPSILON:
                    continue
                if transition[0] not in clousure_transitions:
                    clousure_transitions[transition[0]] = set()
                clousure_transitions[transition[0]].add(transition[1])
        return clousure_transitions
        
        
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
            # print (line)
            lhs, _ = line.replace(" ", "").split("::=")
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
    loop_indent: List[int] = field(default_factory=list)
    if_indent: List[int] = field(default_factory=list)
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
        # print("COMPLETE", state)
        if(state.rule_name in self.states[state.pos]):
            for parent_state in self.states[state.pos][state.rule_name]:
                transitions = self.grammar.NFAs[parent_state.rule_name].GetTransitions(parent_state.node_num)
                for trans in transitions:
                    if trans[0] == state.rule_name:
                        self.complete_times += 1
                        self.queue.append(State(parent_state.rule_name, trans[1], parent_state.pos, self.GetAccepted(parent_state.rule_name, trans[1])))    
    
    def _scan_predict(self, state: State, token: str):
        transitions = self.grammar.NFAs[state.rule_name].GetTransitions(state.node_num)
        for trans in transitions:
            # Scanning.
            if is_terminal(trans[0]):
                if trans[0] == EPSILON:
                    self.scan_times += 1
                    # print(EPSILON, State(state.rule_name, trans[1], state.pos, self.GetAccepted(state.rule_name, trans[1])))
                    self.queue.append(State(state.rule_name, trans[1], state.pos, self.GetAccepted(state.rule_name, trans[1])))
                if ((trans[0] == token) 
                    or (trans[0] == XGRAMMAR_EVERYTHING_FLAG and token != "\\")
                    or (trans[0] == XGRAMMAR_DIGIT_FLAG and token.isdigit())
                    or (trans[0] == XGRAMMAR_HEX_FLAG and token in "0123456789abcdefABCDEF")
                    or (trans[0] == WHITE_SPACE_FLAG and token == " ")
                    or (trans[0] == VARIABLE_FLAG and token in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
                    or (trans[0] ==OR_FLAG and token == "|")):
                    self.scan_times += 1
                    # print("SCAN", State(state.rule_name, trans[1], state.pos, self.GetAccepted(state.rule_name, trans[1])))
                    self.next_states.add(State(state.rule_name, trans[1], state.pos, self.GetAccepted(state.rule_name, trans[1])))
            else:
            # Predicting.
                if(int(trans[0]) not in self.states[len(self.states) - 1]):
                    self.states[len(self.states) - 1][int(trans[0])] = set()
                self.predict_times += 1
                self.states[len(self.states) - 1][int(trans[0])].add(state)
                # print("PREDICT", State(trans[0], 0, len(self.states) - 1, self.GetAccepted(state.rule_name, 0)))
                self.queue.append(State(trans[0], 0, len(self.states) - 1, self.GetAccepted(state.rule_name, 0)))
            
    
    def _consume(self, token: str):
        self.state_num += len(self.current_states)
        self.states.append(dict())
        self.input += token
        self.queue = [s for s in self.current_states]
        assert(len(self.queue) > 0)
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
            # print(token)
            indent = self.CheckIndent(token)
            if indent:
                continue
            if token == "\n":
                self.lines += 1
                self.MarkIndent()
                continue
            self._consume(token)
        print(self.state_num)
        print(self.complete_times)
        print(self.scan_times)
        print(self.predict_times)
        print(self.complete_times + self.scan_times + self.predict_times)
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
                self._scan_predict(state, "EOF")
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
    function_definition ::= 'def' whitespaces variable '(' in_args? ')' ':' FORCE_FLAG COMPLETE_FLAG
    Float ::= Int '.' Int 
    Int ::= DIGIT+
    Array ::= expr '[' expr? ']' | '[]' | '[' in_args ']'
    Dict ::= '{' key_values? '}'
    key_values ::= pair* expr ':' expr 
    pair ::= expr ':' expr ',' 
    String ::= "\"" chars? "\" | '\'' chars? '\''
    expr_statement ::= expr COMPLETE_FLAG
    if_statement ::= 'if' whitespaces expr ':' FORCE_FLAG COMPLETE_FLAG
    else_statement ::= 'else:' COMPLETE_FLAG FORCE_FLAG
    while_statement ::= 'while' whitespaces expr ':' COMPLETE_FLAG FORCE_FLAG
    for_statement ::= 'for' whitespaces variable whitespaces 'in' whitespaces expr ':' COMPLETE_FLAG FORCE_FLAG
    break_statement ::= 'break' COMPLETE_FLAG
    continue_statement ::= 'continue' COMPLETE_FLAG
    return_statement ::= 'return' whitespaces expr? COMPLETE_FLAG
    assign_op ::= '=' | '+=' | '-=' | '*=' | '/=' | '%=' | '&='  | '^=' | '<<=' | '>>=' | '**=' | OR_FLAG '=' | '//='
    expr_binary_op ::= '+' | '-' | '*' | '/' | '%' | '&' | '^' | '<<' | '>>' | '**' | '==' | '!=' | '<' | '>' | '<=' | '>=' | 'and' | 'or' | OR_FLAG | '//' | 'is' | 'in'
    expr_unary_op ::= '+' | '-' | '~' | 'not'
    in_args ::= expr comma_expr* 
    comma_expr ::= ',' expr
    func_call ::= variable '(' in_args? ')'
    expr ::= whitespaces? expr_raw whitespaces? 
    expr_raw ::= Int | Float | String | Bool | variable_raw | func_call | expr expr_binary_op expr | expr_unary_op expr | '(' expr ')' | expr assign_op expr | Array | expr '.' expr | expr ',' expr | Dict
    variable ::= whitespaces? variable_raw whitespaces?
    variable_raw ::= variable_char | variable_raw variable_char | variable_raw DIGIT
    variable_char ::= VARIABLE_FLAG
    chars ::= EVERYTHING | chars EVERYTHING | chars escaped | escaped
    escaped ::= '\\\\' | '\\"'  | '\\n'  | '\\b'  | '\\f'  | '\\r' | '\\t' 
    Bool ::= 'True' | 'False'
    whitespaces ::= WHITE_SPACE_FLAG+
    """
)

# for nfa in grammar.NFAs.values():
#     print(nfa)
# assert(False)
now_time = time.time()
Parser(grammar).read(
    """
def initialize_graph(vertices):
    graph = {}
    i = 1
    while i <= vertices:
        graph[i] = {}
        i = i + 1
    return graph

def add_edge(graph, u, v, capacity):
    if u in graph:
        graph[u][v] = capacity
    if v in graph:
        graph[v][u] = 0

def bfs(graph, source, sink, parent, vertices):
    visited = {}
    i = 1
    while i <= vertices:
        visited[i] = False
        i = i + 1
    queue = [source]
    visited[source] = True
    while len(queue) > 0:
        current = queue.pop(0)
        for neighbor in graph[current]:
            if not visited[neighbor] and graph[current][neighbor] > 0:
                queue.append(neighbor)
                visited[neighbor] = True
                parent[neighbor] = current
                if neighbor == sink:
                    return True
    return False

def edmonds_karp(graph, source, sink, vertices):
    parent = {}
    max_flow = 0
    while bfs(graph, source, sink, parent, vertices):
        path_flow = float('inf')
        current = sink
        while current != source:
            path_flow = min(path_flow, graph[parent[current]][current])
            current = parent[current]
        max_flow = max_flow + path_flow
        current = sink
        while current != source:
            prev = parent[current]
            graph[prev][current] = graph[prev][current] - path_flow
            graph[current][prev] = graph[current][prev] + path_flow
            current = prev
    return max_flow

vertices = 6
graph = initialize_graph(vertices)
add_edge(graph, 1, 2, 16)
add_edge(graph, 1, 3, 13)
add_edge(graph, 2, 3, 10)
add_edge(graph, 2, 4, 12)
add_edge(graph, 3, 2, 4)
add_edge(graph, 3, 5, 14)
add_edge(graph, 4, 3, 9)
add_edge(graph, 4, 6, 20)
add_edge(graph, 5, 4, 7)
add_edge(graph, 5, 6, 4)
source = 1
sink = 6
max_flow = edmonds_karp(graph, source, sink, vertices)
if max_flow > 0:
    print("The maximum possible flow is", max_flow)
else:
    print("No flow is possible from source to sink")
    """
)
print("The time is", time.time() - now_time, "s.")
print()