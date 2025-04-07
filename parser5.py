from dataclasses import dataclass, field
from typing import List, Set, Tuple, Union, Dict
import time

global_rule_dict: dict[str, int] = {}
global_rule_dict_reversed: dict[int, str] = {}
ROOT_RULE= "$"
root_rule_number = 0
INDENT_WB_NUMBER = 4
EPSILON = "EPSILON"
# The following flags are used to represent a universal symbol.
XGRAMMAR_EVERYTHING_FLAG = "EVERYTHING"
XGRAMMAR_HEX_FLAG = "HEX"
XGRAMMAR_DIGIT_FLAG = "DIGIT"
FORCE_FLAG = "FORCE_FLAG"
WHITE_SPACE_FLAG = "WHITE_SPACE_FLAG"
OR_FLAG = "OR_FLAG"
VARIABLE_FLAG = "VARIABLE_FLAG"
NEWLINE = "NEWLINE"
INDENT = "INDENT"
DEDENT = "DEDENT"
ENDMARKER = "ENDMARKER"



def is_terminal(symbol: Union[str, int]) -> bool:
    return isinstance(symbol, str)

def is_universal(symbol: Union[str, int]) -> bool:
    return isinstance(symbol, str) and symbol in {XGRAMMAR_EVERYTHING_FLAG, XGRAMMAR_HEX_FLAG, XGRAMMAR_DIGIT_FLAG, WHITE_SPACE_FLAG, VARIABLE_FLAG, OR_FLAG, NEWLINE, INDENT, DEDENT, ENDMARKER, EPSILON}
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
                if((symbol[0] == '\'' and symbol[len(symbol) - 1] == '\'')
                or (symbol[0] == '"' and symbol[len(symbol) - 1] == '"')):
                    quotation_signal = True
                    symbol = symbol[1:len(symbol) - 1]           
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
            lhs, _ = line.replace(" ", "").split("::=")
            cnt += 1
            if (lhs not in global_rule_dict):
                if lhs == ROOT_RULE:
                    global_rule_dict[lhs] = root_rule_number
                    global_rule_dict_reversed[root_rule_number] = lhs
                else:
                    global_rule_dict[lhs] = cnt
                    global_rule_dict_reversed[cnt] = lhs
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
    def __repr__(self):
       return f"[State: rule_name={global_rule_dict_reversed[self.rule_name]}, node_num={self.node_num}, pos={self.pos}, accepted={self.accepted}]"

@dataclass
class Parser:
    grammar: Grammar
    now_indent: int = 0
    state_num = 0
    tokens_num = 0
    tokens_num_without_indent = 0
    lines = 0
    complete_times = 0
    scan_times = 0
    predict_times = 0
    init_times = 0
    line_start: bool = True
    
    def Inline(self):
        for nfa in self.grammar.NFAs.values():
            new_dict = {}
            for transitions in nfa.transitions.values():
                queue = []
                new_queue = []
                for transition in transitions:
                    if not is_terminal(transition[0]):
                        to_inline_nfa = self.grammar.NFAs[transition[0]]
                        # The NFA is easy enough to be inlined.
                        if not to_inline_nfa.easy:
                            continue
                        
                        # The first transition is enter the rule.
                        # The second transition is the returning rule.
                        new_transition = (EPSILON, nfa.node_cnt)
                        queue.append(transition)
                        new_queue.append(new_transition)
                        
                        for inlined_transitions_raw in to_inline_nfa.transitions:
                            inlined_transitions = inlined_transitions_raw + nfa.node_cnt
                            new_dict[inlined_transitions] = set()
                            for inline_trans in to_inline_nfa.transitions[inlined_transitions_raw]:
                                new_dict[inlined_transitions].add((inline_trans[0], inline_trans[1] + nfa.node_cnt))
                        for final_node in to_inline_nfa.final_node:
                            if not (final_node + nfa.node_cnt) in new_dict:
                                new_dict[final_node + nfa.node_cnt] = set()
                            new_dict[final_node + nfa.node_cnt].add((EPSILON, transition[1]))
                        nfa.node_cnt += to_inline_nfa.node_cnt
                while queue:
                    transitions.remove(queue.pop(0))
                    transitions.add(new_queue.pop(0))
            for new_transitions in new_dict:
                if not new_transitions in nfa.transitions:
                    nfa.transitions[new_transitions] = new_dict[new_transitions]
                else:
                    nfa.transitions[new_transitions] = nfa.transitions[new_transitions].union(new_dict[new_transitions])
    def __post_init__(self):
        self.Inline()
        for nfa in self.grammar.NFAs.values():
            nfa.ToDFA()
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
                    or (trans[0] == XGRAMMAR_EVERYTHING_FLAG and token != "\\" and token != "\n" and token != "INDENT" and token != "DEDENT")
                    or (trans[0] == XGRAMMAR_DIGIT_FLAG and token.isdigit())
                    or (trans[0] == XGRAMMAR_HEX_FLAG and token in "0123456789abcdefABCDEF")
                    or (trans[0] == WHITE_SPACE_FLAG and token == " ")
                    or (trans[0] == VARIABLE_FLAG and token in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
                    or (trans[0] == OR_FLAG and token == "|")
                    or (trans[0] == NEWLINE and token == "\n")):
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
        self.queue = [s for s in self.current_states]
        assert(len(self.queue) > 0)
        self.current_states.clear()
        while self.queue:
            state = self.queue.pop(0)
            # print(state)
            if state in self.current_states:
                continue
            self.current_states.add(state)
            if state.accepted:
                self._complete(state)
            self._scan_predict(state, token)
        self.current_states = self.next_states
        # print("__________")
        self.next_states = set()
    
    def read(self, text: str):
        tmp_indent = 0
        for token in text:
            # print(self.input)
            self.input += token
            if self.line_start and token == " ":
                tmp_indent += 1
                continue
            # The line isn't empty.
            if token == "\n":
                if not self.line_start:
                    self.lines += 1
                    self.line_start = True
                    self.now_indent = tmp_indent
                    tmp_indent = 0
                    self._consume("\n")
                continue
            # The first token which isn't a space, and now check the indentation.
            if self.line_start:
                self.line_start = False
                if tmp_indent % INDENT_WB_NUMBER != 0:
                    raise ValueError("The indentation is not correct.")
                if tmp_indent > self.now_indent:
                    depth = (tmp_indent - self.now_indent) // INDENT_WB_NUMBER
                    for i in range(0, depth):
                        self._consume("INDENT")
                elif tmp_indent < self.now_indent:
                    depth = (self.now_indent - tmp_indent) // INDENT_WB_NUMBER
                    for i in range(0, depth):
                        self._consume("DEDENT")
                
            self._consume(token)
        print(self.state_num)
        print(self.complete_times)
        print(self.scan_times)
        print(self.predict_times)
        print(self.complete_times + self.scan_times + self.predict_times)
        return self
        
        
    def GetAccepted(self, rule:int, node:int) -> bool:
        return self.grammar.NFAs[rule].Accepted(node)
    
    def Accepted(self) -> bool:
        for state in self.current_states:
            if state.rule_name == root_rule_number and state.accepted:
                return True
        return False
    

    def _print(self):
        assert(len(self.current_states) > 0)
        # print(self.input ,len(self.current_states) > 0)



grammar = Grammar.parse(
    """
$ ::= file
whitespaces ::= WHITE_SPACE_FLAG+
file ::= statements? ENDMARKER 
interactive ::= statement_newline 
eval ::= expressions NEWLINE* ENDMARKER 
func_type ::= '(' whitespaces? type_expressions? whitespaces? ')' whitespaces? '->' whitespaces? expression NEWLINE* ENDMARKER 
statements ::= statement+ 
statement ::= compound_stmt  | simple_stmts 
statement_newline ::= compound_stmt NEWLINE | simple_stmts | NEWLINE | ENDMARKER 
sep_simle_stmt ::= whitespaces? ';' whitespaces? simple_stmt
simple_stmts ::= simple_stmt sep_simle_stmt* ';'? NEWLINE 
simple_stmt ::= assignment | type_alias | star_expressions | return_stmt | import_stmt | raise_stmt
simple_stmt ::= 'pass' | del_stmt | yield_stmt | assert_stmt | 'break'  | 'continue'  | global_stmt | nonlocal_stmt
compound_stmt ::= function_def | if_stmt | class_def | with_stmt | for_stmt | try_stmt | while_stmt | match_stmt
=annotated_rhs ::= '=' annotated_rhs
assignment ::= NAME ':' expression =annotated_rhs?
assignment ::= '(' single_target ')' ':' expression =annotated_rhs? 
assignment ::= single_subscript_attribute_target ':' expression =annotated_rhs?
starexp_eq ::= star_targets '='
assignment ::= starexp_eq+ yield_expr | starexp_eq+ star_expressions
assignment ::= single_target whitespaces? augassign whitespaces? yield_expr | single_target whitespaces? augassign whitespaces? star_expressions 
annotated_rhs ::= yield_expr | star_expressions
augassign ::= '=' | '+=' | '-=' | '*=' | '@=' | '/=' | '%=' | '&=' | OR_FLAG '=' | '^=' | '<<=' | '>>=' | '**=' | '//=' 
return_stmt ::= 'return' whitespaces star_expressions? | 'return'
raise_stmt ::= 'raise' expression 'from' expression | 'raise' | 'raise' expression
del_stmt ::= 'del' whitespaces del_targets  
comma_name ::= ',' whitespaces? NAME
global_stmt ::= 'global' whitespaces NAME whitespaces? comma_name*
nonlocal_stmt ::= 'nonlocal' whitespaces NAME whitespaces? comma_name*
yield_stmt ::= yield_expr 
assert_stmt ::= 'assert' whitespaces expression whitespaces? ',' whitespaces? expression | 'assert' whitespaces expression
import_stmt ::= import_name | import_from
import_name ::= 'import' whitespaces dotted_as_names 
dot_or_three_dots ::= '.' | '...'
import_from ::= 'from' whitespaces dot_or_three_dots* whitespaces dotted_name whitespaces 'import' whitespaces import_from_targets 
import_from ::= 'from' whitespaces dot_or_three_dots+ whitespaces 'import' whitespaces import_from_targets 
import_from_targets ::= '(' whitespaces? import_from_as_names whitespaces? ','? whitespaces? ')'  | '*' 
import_from_targets ::= import_from_as_names
comma_as_name ::= ',' import_from_as_name
import_from_as_names ::= import_from_as_name comma_as_name*
import_from_as_name ::= NAME whitespaces 'as' whitespaces NAME | NAME
comma_dotted_as_name ::= ',' whitespaces? dotted_as_name
dotted_as_names ::= dotted_as_name whitespaces? comma_dotted_as_name*
as_name ::=  whitespaces 'as' whitespaces NAME
dotted_as_name ::= dotted_name as_name?
dotted_name ::= dotted_name '.' NAME | NAME
block ::= NEWLINE INDENT statements DEDENT | simple_stmts
decorator_inner ::= '@' named_expression NEWLINE
decorators ::= decorator_inner+
class_def ::= decorators? class_def_raw 
param_arguments ::= '(' whitespaces? arguments? whitespaces? ')'
class_def_raw ::= 'class' NAME type_params? param_arguments? ':' block 
function_def ::= decorators? function_def_raw 
arrow_exp ::= whitespaces? '->' whitespaces? expression
function_def_raw ::= 'def' whitespaces NAME type_params? '(' params? ')' arrow_exp? ':' func_type_comment? block 
function_def_raw ::= 'async' whitespaces 'def' whitespaces NAME type_params? '(' params? ')' arrow_exp?  ':' func_type_comment? block 
params ::= parameters
parameters ::= slash_no_default param_no_default* param_with_default* star_etc? | slash_with_default param_with_default* star_etc? 
parameters ::= param_no_default+ param_with_default* star_etc? | param_with_default+ star_etc? | star_etc 
slash_no_default ::= param_no_default+ whitespaces? '/' whitespaces? ',' 
slash_no_default ::= param_no_default+ whitespaces? '/'
slash_with_default ::= param_no_default* param_with_default+ whitespaces? '/' whitespaces? ',' 
slash_with_default ::=  param_no_default* param_with_default+ whitespaces? '/'
star_etc ::= '*' param_no_default param_maybe_default* kwds? | '*' param_no_default_star_annotation param_maybe_default* kwds? 
star_etc ::= '*' ',' param_maybe_default+ kwds? | kwds 
kwds ::= | '**' param_no_default 
param_no_default ::= whitespaces? param whitespaces? ',' TYPE_COMMENT? 
param_no_default ::= whitespaces? param whitespaces? TYPE_COMMENT?
param_no_default_star_annotation ::= param_star_annotation ',' TYPE_COMMENT?  
param_no_default_star_annotation ::= param_star_annotation TYPE_COMMENT?
param_with_default ::= param default ',' TYPE_COMMENT? 
param_with_default ::= param default TYPE_COMMENT?
param_maybe_default ::= param default? ',' TYPE_COMMENT? 
param_maybe_default ::= param default? TYPE_COMMENT? 
param ::= NAME annotation? 
param_star_annotation ::= NAME star_annotation 
annotation ::= ':' expression 
star_annotation ::= ':' star_expression 
default ::= '=' expression 
if_stmt ::= 'if' whitespaces named_expression whitespaces? ':' block elif_stmt | 'if' whitespaces named_expression whitespaces? ':' block else_block?
elif_stmt ::= 'elif' whitespaces named_expression whitespaces? ':' block elif_stmt | 'elif' whitespaces named_expression whitespaces? ':' block else_block? 
else_block ::= 'else' whitespaces? ':' block 
while_stmt ::= 'while' whitespaces named_expression whitespaces? ':' block else_block? 
for_stmt ::= 'for' whitespaces star_targets whitespaces 'in' whitespaces star_expressions whitespaces? ':' TYPE_COMMENT? block else_block? 
for_stmt ::= 'async' whitespaces 'for' whitespaces star_targets whitespaces 'in' whitespaces star_expressions whitespaces? ':' TYPE_COMMENT? block else_block? 
comma_with_item ::= ',' with_item
with_inner ::= with_item comma_with_item*
with_stmt ::= 'with' whitespaces '(' whitespaces? with_inner whitespaces? ','? whitespaces? ')' whitespaces? ':' TYPE_COMMENT? block | 'with' whitespaces with_inner whitespaces? ':' TYPE_COMMENT? block 
with_stmt ::= 'async' whitespaces 'with' whitespaces '(' whitespaces? with_inner whitespaces? ','? whitespaces? ')' whitespaces? ':' block | 'async' whitespaces 'with' whitespaces with_item comma_with_item* whitespaces? ':' TYPE_COMMENT? block 
with_item ::= expression whitespaces 'as' whitespaces star_target | expression 
try_stmt ::= 'try' whitespaces? ':' block finally_block | 'try' whitespaces? ':' block except_block+ else_block? finally_block? 
try_stmt ::= 'try' whitespaces? ':' block except_star_block+ else_block? finally_block?
except_block ::= 'except' whitespaces expression as_name? whitespaces? ':' block | 'except' whitespaces? ':' block 
except_star_block ::= 'except' '*' expression as_name? ':' block 
finally_block ::= 'finally' ':' block 
match_stmt ::= 'match' subject_expr ':' NEWLINE INDENT case_block+ DEDENT 
subject_expr ::= star_named_expression ',' star_named_expressions? | named_expression
case_block ::= 'case' patterns guard? ':' block 
guard ::= 'if' named_expression 
patterns ::= open_sequence_pattern | pattern
pattern ::= as_pattern | or_pattern
as_pattern ::= or_pattern 'as' pattern_capture_target 
OR_closed_pattern ::= OR_FLAG closed_pattern
or_pattern ::= closed_pattern OR_closed_pattern*
closed_pattern ::= literal_pattern | capture_pattern | wildcard_pattern
closed_pattern ::= value_pattern | group_pattern | sequence_pattern | mapping_pattern | class_pattern
literal_pattern ::= signed_number  | complex_number | strings | 'None' | 'True' | 'False' 
literal_expr ::= signed_number | complex_number | strings | 'None' | 'True' | 'False' 
complex_number ::= signed_real_number '+' imaginary_number | signed_real_number '-' imaginary_number  
signed_number ::= NUMBER |'-' NUMBER 
signed_real_number ::= real_number | '-' real_number 
real_number ::= NUMBER 
imaginary_number ::= NUMBER 
capture_pattern ::= pattern_capture_target 
pattern_capture_target ::= EPSILON
wildcard_pattern ::= | "_" 
value_pattern ::= attr
attr ::= name_or_attr '.' NAME 
name_or_attr ::= attr | NAME
group_pattern ::= '(' pattern ')' 
sequence_pattern ::= '[' maybe_sequence_pattern? ']' | '(' open_sequence_pattern? ')' 
open_sequence_pattern ::= maybe_star_pattern ',' maybe_sequence_pattern? 
comma_maybe_star_pattern ::= maybe_star_pattern ','
maybe_sequence_pattern ::= maybe_star_pattern comma_maybe_star_pattern* ','? 
maybe_star_pattern ::= star_pattern | pattern
star_pattern ::= '*' pattern_capture_target | '*' wildcard_pattern 
mapping_pattern ::= '{' '}' | '{' double_star_pattern ','? '}' 
mapping_pattern ::= '{' items_pattern ',' double_star_pattern ','? '}' | '{' items_pattern ','? '}' 
comma_kv_pattern ::= ',' key_value_pattern
items_pattern ::= key_value_pattern comma_kv_pattern*
key_value_pattern ::= literal_expr ':' pattern | attr ':' pattern 
double_star_pattern ::= '**' pattern_capture_target 
class_pattern ::= name_or_attr '(' ')' | name_or_attr '(' positional_patterns ','? ')' 
class_pattern ::= name_or_attr '(' keyword_patterns ','? ')' | name_or_attr '(' positional_patterns ',' keyword_patterns ','? ')' 
comma_pattern ::= ',' pattern
positional_patterns ::= pattern comma_pattern*
comma_keyword_pattern ::= ',' keyword_pattern
keyword_patterns ::= keyword_pattern comma_keyword_pattern*
keyword_pattern ::= NAME '=' pattern 
type_alias ::= 'type' NAME type_params? '=' expression 
type_params ::=  '[' type_param_seq ']' 
comma_type_param ::= ',' type_param
type_param_seq ::= type_param comma_type_param* ','? 
type_param ::= NAME type_param_bound? type_param_default?
type_param ::= '*' NAME type_param_starred_default? 
type_param ::= '**' NAME type_param_default?
type_param_bound ::= ':' expression 
type_param_default ::= '=' expression 
type_param_starred_default ::= '=' star_expression 
comma_exp ::= ',' expression
expressions ::= expression comma_exp+ ','? | expression ',' | expression
expression ::= disjunction whitespaces 'if' whitespaces disjunction whitespaces 'else' whitespaces expression | disjunction | lambdef
yield_expr ::= 'yield' whitespaces 'from' expression | 'yield' star_expressions? 
comma_star_exp ::= ',' star_expression
star_expressions ::= star_expression comma_star_exp+ ','? | star_expression ',' | star_expression
star_expression ::= '*' bitwise_or | expression
comma_star_named_exp ::= ',' star_named_expression
star_named_expressions ::= star_named_expression comma_star_exp* ','? 
star_named_expression ::= '*' bitwise_or | named_expression
assignment_expression ::= NAME  whitespaces? ':=' whitespaces? expression 
named_expression ::= assignment_expression | expression
or_conjunction ::= 'or' whitespaces conjunction
disjunction ::= conjunction or_conjunction* 
and_inversion ::=  whitespaces 'and' whitespaces inversion
conjunction ::= inversion and_inversion* 
inversion ::= 'not' whitespaces inversion | comparison
comparison ::= bitwise_or compare_op_bitwise_or_pair+ | bitwise_or
compare_op_bitwise_or_pair ::= eq_bitwise_or | noteq_bitwise_or | lte_bitwise_or
compare_op_bitwise_or_pair ::= lt_bitwise_or | gte_bitwise_or | gt_bitwise_or
compare_op_bitwise_or_pair ::= notin_bitwise_or | in_bitwise_or | isnot_bitwise_or | is_bitwise_or
eq_bitwise_or ::= whitespaces? '==' whitespaces? bitwise_or 
noteq_bitwise_or ::= whitespaces? '!=' whitespaces? bitwise_or 
lte_bitwise_or ::= whitespaces? '<=' whitespaces? bitwise_or 
lt_bitwise_or ::=  whitespaces? '<' whitespaces? bitwise_or 
gte_bitwise_or ::= whitespaces? '>=' whitespaces? bitwise_or 
gt_bitwise_or ::= whitespaces? '>' whitespaces? bitwise_or 
notin_bitwise_or ::= 'not' 'in' bitwise_or 
in_bitwise_or ::= whitespaces 'in' whitespaces bitwise_or 
isnot_bitwise_or ::= 'is' 'not' bitwise_or 
is_bitwise_or ::= 'is' bitwise_or 
bitwise_or ::= bitwise_or OR_FLAG bitwise_xor | bitwise_xor
bitwise_xor ::= bitwise_xor '^' bitwise_and | bitwise_and
bitwise_and ::= bitwise_and '&' shift_expr | shift_expr
shift_expr ::= shift_expr '<<' sum | shift_expr '>>' sum | sum
sum ::= sum whitespaces? '+' whitespaces? term | sum whitespaces? '-' whitespaces? term | term
term ::= term whitespaces? '*' whitespaces? factor | term whitespaces? '/' whitespaces? factor | term whitespaces? '//' whitespaces? factor | term whitespaces? '%' whitespaces? factor 
term ::= term whitespaces? '@' whitespaces? factor | factor
factor ::= '+' factor | '-' factor | '~' factor | power
power ::= await_primary '**' factor | await_primary
await_primary ::= 'await' primary | primary
primary ::= primary '.' NAME | primary genexp | primary '(' arguments? ')' 
primary ::= primary '[' slices ']' | atom
comma_slice_or_stared_exp ::= ',' slice | ',' starred_expression
slices ::=  slice | comma_slice_or_stared_exp+ ','? 
slice ::= expression? ':' expression? | named_expression | expression? ':' expression? ':' expression?
atom ::= NAME | 'True' | 'False' | 'None' | strings | NUMBER | tuple | group | genexp
atom ::= list | listcomp | dict | set | dictcomp | setcomp | '...' 
group ::= '(' yield_expr ')' | '(' named_expression ')' 
lambdef ::= 'lambda' lambda_params? ':' expression 
lambda_params ::= lambda_parameters
lambda_parameters ::= lambda_slash_no_default lambda_param_no_default* lambda_param_with_default* lambda_star_etc? 
lambda_parameters ::= lambda_slash_with_default lambda_param_with_default* lambda_star_etc? 
lambda_parameters ::= lambda_param_no_default+ lambda_param_with_default* lambda_star_etc? 
lambda_parameters ::= lambda_param_with_default+ lambda_star_etc? | lambda_star_etc 
lambda_slash_no_default ::= lambda_param_no_default+ '/' ',' | lambda_param_no_default+ '/'
lambda_slash_with_default ::= lambda_param_no_default* lambda_param_with_default+ '/' ',' 
lambda_slash_with_default ::= lambda_param_no_default* lambda_param_with_default+ '/'
lambda_star_etc ::= '*' lambda_param_no_default lambda_param_maybe_default* lambda_kwds?
lambda_star_etc ::= '*' ',' lambda_param_maybe_default+ lambda_kwds? | lambda_kwds 
lambda_kwds ::= '**' lambda_param_no_default 
lambda_param_no_default ::= lambda_param ','? 
lambda_param_with_default ::= lambda_param default ','? 
lambda_param_maybe_default ::= lambda_param default? ','?
lambda_param ::= NAME 
fstring_middle ::= fstring_replacement_field | FSTRING_MIDDLE 
fstring_replacement_field ::= '{' annotated_rhs '='? fstring_conversion? fstring_full_format_spec '}' 
fstring_conversion ::= '!' NAME 
fstring_full_format_spec ::= ':' fstring_format_spec* 
fstring_format_spec ::= FSTRING_MIDDLE | fstring_replacement_field
fstring ::= FSTRING_START fstring_middle* FSTRING_END 
string ::= STRING 
fstring_or_string ::= fstring | string
strings ::= fstring_or_string+ 
list ::= '[' star_named_expressions? ']' 
tuple_inner ::= star_named_expression ',' star_named_expression?
tuple ::= '(' tuple_inner? ')' 
set ::= '{' star_named_expressions '}' 
dict ::= '{' double_starred_kvpairs? '}'
comma_double_starred_kvpair ::= ',' double_starred_kvpair 
double_starred_kvpairs ::= double_starred_kvpair comma_double_starred_kvpair? ','? 
double_starred_kvpair ::= '**' bitwise_or | kvpair
kvpair ::= expression ':' expression 
for_if_clauses ::= for_if_clause+ 
if_disjunction ::= 'if' whitespaces disjunction
for_if_clause ::= 'async' whitespaces 'for' whitespaces star_targets whitespaces 'in' whitespaces disjunction if_disjunction* 
for_if_clause ::= 'for' whitespaces star_targets whitespaces 'in' whitespaces disjunction if_disjunction* 
listcomp ::= '[' named_expression for_if_clauses ']' 
setcomp ::= '{' named_expression for_if_clauses '}' 
genexp ::= '(' assignment_expression for_if_clauses ')' 
genexp ::= '(' expression for_if_clauses ')' 
dictcomp ::= '{' kvpair for_if_clauses '}' 
arguments ::= args whitespaces? ','?
stared_exp_or_assign_exp_or_exp ::= starred_expression | assignment_expression | expression
comma_sae ::= ',' whitespaces? stared_exp_or_assign_exp_or_exp 
comma_kwargs ::= ',' kwargs
args ::= stared_exp_or_assign_exp_or_exp comma_sae* comma_kwargs? | kwargs 
comma_kwarg_or_starred ::= ',' kwarg_or_starred
comma_kwarg_or_double_starred ::= ',' kwarg_or_double_starred
kwargs ::= kwarg_or_starred comma_kwarg_or_starred* kwarg_or_double_starred comma_kwarg_or_double_starred*
kwargs ::= kwarg_or_starred comma_kwarg_or_starred* | kwarg_or_double_starred comma_kwarg_or_double_starred*
starred_expression ::= '*' expression 
kwarg_or_starred ::= NAME '=' expression | starred_expression 
kwarg_or_double_starred ::= NAME '=' expression | '**' expression
comma_star_target ::= ',' star_target 
star_targets ::= star_target comma_star_target* ','? 
star_targets_list_seq ::= star_target comma_star_target* ','? 
star_targets_tuple_seq ::= star_target comma_star_target+ ','? | star_target ',' 
star_target ::= '*' star_target | target_with_star_atom
target_with_star_atom ::= t_primary '.' NAME | t_primary '[' slices ']' | star_atom
star_atom ::= NAME | '(' target_with_star_atom ')' | '(' star_targets_tuple_seq? ')' 
star_atom ::= '[' star_targets_list_seq? ']' 
single_target ::= single_subscript_attribute_target | NAME | '(' single_target ')' 
single_subscript_attribute_target ::= t_primary '.' NAME | t_primary '[' slices ']' 
t_primary ::= t_primary '.' NAME | t_primary '[' slices ']' | t_primary genexp 
t_primary ::= t_primary '(' arguments? ')' | atom
t_lookahead ::= '(' | '[' | '.'
comma_del_target ::= ',' del_target
del_targets ::= del_target comma_del_target* ','?
del_target ::= t_primary '.' NAME | t_primary '[' slices ']' | del_t_atom
del_t_atom ::= NAME | '(' del_target ')' | '(' del_targets? ')' | '[' del_targets? ']' 
type_expressions::= expression comma_exp* ',' '*' expression ',' '**' expression 
type_expressions::= expression comma_exp* ',' '*' expression 
type_expressions::= expression comma_exp* ',' '**' expression 
type_expressions::='*' expression ',' '**' expression 
type_expressions::= '*' expression 
type_expressions::= '**' expression 
type_expressions::= expression comma_exp*
func_type_comment ::= NEWLINE TYPE_COMMENT | TYPE_COMMENT
NAME ::= NAME VARIABLE_FLAG | NAME DIGIT | VARIABLE_FLAG
TYPE_COMMENT ::= '#' EVERYTHING+
NUMBER ::= INT | FLOAT
INT ::= DIGIT+
FLOAT ::= INT '.' INT
FSTRING_MIDDLE ::= EVERYTHING
FSTRING_START ::= 'f"' | 'f'''
FSTRING_END ::= '"' | '''
STRING ::= '"' chars '"' | ''' chars '''
chars ::= EVERYTHING | chars EVERYTHING | chars escaped | escaped
escaped ::= '\\\\' | '\\"'  | '\\n'  | '\\b'  | '\\f'  | '\\r' | '\\t' 
    """
)
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