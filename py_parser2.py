from dataclasses import dataclass
from typing import List, Set, Tuple, Union
import time
# Modified based on DarkSharpness's code

ROOT_RULE= "$"
root_rule_number = 0

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

# We use int to represent non-terminal symbols and str to represent terminal symbols.
def is_terminal(symbol: Union[str, int]) -> str | None:
    if isinstance(symbol, int):
        return None
    return symbol

def is_nonterminal(symbol: Union[str, int]) -> int | None:
    if isinstance(symbol, str):
        return None
    return symbol


@dataclass(frozen=True)
class Grammar:
    rules: Tuple[Tuple[int, str], ...]
    def parse(rule: str) -> "Grammar":
        global loop_rules
        global def_rules
        global force_rules
        global need_def_rules
        global need_loop_rules
        global if_rules
        global need_if_rules
        global complete_line_rules
        rule_dict = {str:int}
        cnt = 0
        results = []
        # Do the first scanning. Add all the non-terminal symbols to the dictionary.
        for line in rule.split("\n"):
            if not line.strip():
                continue
            lhs, rhs = line.replace(" ", "").split("::=")
            cnt += 1
            if not lhs in rule_dict:
                rule_dict[lhs] = cnt
        # Do the second scanning. Replace the non-terminal symbols with the corresponding number.
        cnt = 0
        for line in rule.split("\n"):
            if not line.strip():
                continue
            cnt = cnt + 1
            lhs, rhs = line.split("::=")
            lhs = lhs.replace(" ", "")
            if(lhs == ROOT_RULE):
                global root_rule_number
                root_rule_number = rule_dict[lhs]
            for rule in rhs.split("|"):
                processed = []
                for symbol in rule.split(" "):
                    # If the symbol is a loop flag, add the rule to the loop_rules set.
                    if(symbol == LOOP_FLAG):
                        loop_rules.add(rule_dict[lhs])
                        continue
                    if(symbol == DEF_FLAG):
                        def_rules.add(rule_dict[lhs])
                        continue
                    if(symbol == FORCE_FLAG):
                        force_rules.add(rule_dict[lhs])
                        continue
                    if(symbol == NEED_LOOP_FLAG):
                        need_loop_rules.add(rule_dict[lhs])
                        continue
                    if(symbol == NEED_DEF_FLAG):
                        need_def_rules.add(rule_dict[lhs])
                        continue
                    if(symbol == IF_FLAG):
                        if_rules.add(rule_dict[lhs])
                        continue
                    if(symbol == NEED_IF_FLAG):
                        need_if_rules.add(rule_dict[lhs])
                        continue
                    if(symbol == COMPLETE_FLAG):
                        complete_line_rules.add(rule_dict[lhs])
                        processed.append("\n")
                        continue
                    if(symbol == ""):
                        continue
                    if symbol in rule_dict:
                        processed.append(rule_dict[symbol])
                    else:
                        processed.append(symbol)    
                if(processed != []):
                    results += [(cnt, processed)]
        global global_rule_dict
        global_rule_dict = rule_dict
        return Grammar(tuple(results))

    def __getitem__(self, symbol: str) -> List[Tuple[int, str]]:
        return [r for r in self.rules if r[0] == symbol]

@dataclass(frozen=True)
class State:
    name: int
    expr: list[Union[int, str]]
    pos: int = 0
    start: int = 0

    def terminated(self) -> bool:
        return self.pos >= len(self.expr)

    def symbol(self) -> str | None:
        return None if self.terminated() else self.expr[self.pos]

    def nonterminal_symbol(self) ->int | None:
        if sym := self.symbol():
            return is_nonterminal(sym)
        return None

    def __next__(self) -> "State":
        return State(self.name, self.expr, self.pos + 1, self.start)

    def __repr__(self) -> str:
        return f'[{self.start}] {self.name} -> ' + \
            f'{self.expr[:self.pos]}•{self.expr[self.pos:]}'
    def __hash__(self):
        return hash((self.name, tuple(self.expr), self.pos, self.start))
END_SYMBOL = "."

@dataclass
class Parser:
    grammar: Grammar
    # state_stack is used to store the now indent and the corresponding state.
    # for example, in the demo, we need to store the state such as for-sentence.
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
        self.state_set: List[Set[State]] = [set()]
        self.inputs: str = ""
        self.state_set[0].add(State(*self.grammar[root_rule_number][0]))
        
    def _complete(self, state: State) -> List[State]:
        results: List[State] = []
        for r in self.state_set[state.start]:
            self.complete_times += 1
            if state.name == r.symbol():
                results.append(next(r))
        return results

    def _predict(self, start: int, symbol: str) -> List[State]:
        self.predict_times +=  len(self.grammar[symbol])
        return [
            State(*r, start=start)
            for r in self.grammar[symbol]
        ]

    def _scan(self, state: State, start: int, token: str):
        self.scan_times += 1
        if state.symbol() == token \
        or (state.symbol() == XGRAMMAR_EVERYTHING_FLAG and state.symbol() != "\\") \
        or (state.symbol() == WHITE_SPACE_FLAG and token == " ") \
        or (state.symbol() == VARIABLE_FLAG \
            and token in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_") \
        or (state.symbol() == OR_FLAG and token == "|")\
        or (state.symbol() == XGRAMMAR_DIGIT_FLAG and token in "0123456789"):
            self.state_set[start + 1].add(next(state))

    def _consume(self, text: str):      
        terminal = is_terminal(text)
        assert terminal is not None and len(terminal) == 1
        self.inputs += terminal

        queue = list(self.state_set.pop())
        new_set = set()
        cur_pos = len(self.state_set)
        self.state_set.append(new_set)
        self.state_set.append(set())
        while queue:
            state = queue.pop(0)
            if state in new_set:
                continue
            new_set.add(state)
            if state.terminated():
                queue += self._complete(state)
            elif nt := state.nonterminal_symbol():
                queue += self._predict(cur_pos, nt)
            else:
                self._scan(state, cur_pos, terminal)
        # If the text is a new line.
        self.state_num += len(self.state_set[-2])
        self.tokens_num += 1
        self.tokens_num_without_indent += 1
        if(text == "\n"):
            self.lines += 1
            for state in self.state_set:
                for s in state:
                    if s.terminated() and s.name in complete_line_rules:
                        #Detect the loop.
                        if force_rules.__contains__(s.name):
                            self.force_indent = True
                        else:
                            self.force_indent = False
                        if loop_rules.__contains__(s.name):
                            self.loop_indent.append(self.now_indent)
                        if if_rules.__contains__(s.name):
                            self.if_indent.append(self.now_indent)
                        #To check the new lines.
                        if def_rules.__contains__(s.name):
                            self.define = True
                        if need_loop_rules.__contains__(s.name):
                            if self.loop_indent == []:
                                raise Exception("Indentation Error: Need Loop")
                        if need_def_rules.__contains__(s.name):
                            if not self.define:
                                raise Exception("Indentation Error: Need Define")
                        if need_if_rules.__contains__(s.name):
                            if not self.now_indent in self.if_indent:
                                raise Exception("Indentation Error: If else Error")
                        self.__post_init__()
                        return
                
            raise Exception("Syntax Error")                  

    def _finalize(self, pos: int):
        queue = list(self.state_set.pop())
        new_set = set()
        cur_pos = len(self.state_set)
        self.state_set.append(new_set)
        while queue:
            state = queue.pop(0)
            if state in new_set:
                continue
            new_set.add(state)
            if state.terminated():
                queue += self._complete(state)
            elif nt := state.nonterminal_symbol():
                queue += self._predict(cur_pos, nt)
        text = self.inputs
        if pos == 1:
            pos = 0
        for i, state in enumerate(self.state_set):
            if i < pos:
                continue
            # accept = any(s.name == root_rule_number and s.terminzzated() for s in state)
            # print(f"State {i}: {text[:i]}•{text[i:]} {accept=}")
            print("\n".join(f"  {s}" for s in state))
            for s in state:
                if s.name == root_rule_number and s.terminated():
                    print(s)
        print("--------------------")
                   

    def _print(self, pos: int) -> None:
        copy = Parser(self.grammar, self.loop_indent, self.if_indent)
        copy.state_set = self.state_set + []
        copy.inputs = self.inputs + ""
        return copy._finalize(pos)

    def read(self, text: str):
        length = len(self.state_set)
        for token in text:
            print("The token is", token)
            
            if token == " " and self.line_start:
                self.tokens_num += 1
                self.white_space_cnt += 1 
                continue
            
            if token == "\n":
                # An empty line.
                if(self.line_start):
                    continue
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
                          
            self._consume(token)
        self._print(length)
        # print("The number of states is", self.state_num)
        # print("The number of tokens is", self.tokens_num)
        # print("The number of lines is", self.lines)
        # print("The number of tokens without indent is", self.tokens_num_without_indent)
        # print("The number of complete lines is", self.complete_times)
        # print("The number of scan times is", self.scan_times)
        # print("The number of predict times is", self.predict_times)
        # print("The average line tokens without indent is", self.tokens_num_without_indent / self.lines)
        print("The ratio of states to tokens without indents is", self.state_num / self.tokens_num_without_indent)
        # print("The ratio of complete times to tokens without indents is", self.complete_times / self.tokens_num_without_indent)
        # print("The ratio of scan times to tokens without indents is", self.scan_times / self.tokens_num_without_indent)
        # print("The ratio of predict times to tokens without indents is", self.predict_times / self.tokens_num_without_indent)
        # print("The line average of the states is", self.state_num / self.lines)
        # print("The line average of complete times is", self.complete_times / self.lines)
        # print("The line average of scan times is", self.scan_times / self.lines)
        # print("The line average of predict times is", self.predict_times / self.lines)
        return self
# In my realization, $ shouldn't have multiple rules. i.e.
# $ ::= Array | Object is undefined.
# If we want to use a rule to parse a "word", 
# we need to use whitespace to separate them.
# XGRAMMAR_EVERYTHING_FLAG = "EVERYTHING"
# XGRAMMAR_HEX_FLAG = "HEX"
# LOOP_FLAG = "LOOP_FLAG"
# DEF_FLAG = "DEF_FLAG"
# FORCE_FLAG = "FORCE_FLAG"
# NEED_LOOP_FLAG = "NEED_LOOP_FLAG"
# NEED_DEF_FLAG = "NEED_DEF_FLAG"
# IF_FLAG = "IF_FLAG"
# NEED_IF_FLAG = "NEED_IF_FLAG"
# COMPLETE_FLAG = "COMPLETE_FLAG"
# WHITE_SPACE_FLAG = "WHITE_SPACE_FLAG"
# OR_FLAG = "OR_FLAG"
# VARIABLE_FLAG = "VARIABLE_FLAG"

grammar = Grammar.parse(
    """
    $ ::= python
    python ::= if_statement | else_statement | while_statement | for_statement | break_statement | continue_statement | return_statement | function_definition | expr_statement
    function_definition ::= d v ( args ) : FORCE_FLAG DEF_FLAG COMPLETE_FLAG | d e f v ( ) : FORCE_FLAG DEF_FLAG COMPLETE_FLAG 
    args ::= args , v | v
    Array ::= expr [ expr ] | [ ] | [ in_args ]
    Dict ::= { } | { key_values }
    key_values ::= key_values , expr : expr | expr : expr
    expr_statement ::= expr COMPLETE_FLAG
    if_statement ::= i f expr : FORCE_FLAG IF_FLAG COMPLETE_FLAG
    else_statement ::= e l s e : NEED_IF_FLAG COMPLETE_FLAG FORCE_FLAG
    while_statement ::= w expr : LOOP_FLAG COMPLETE_FLAG FORCE_FLAG
    for_statement ::= f o r v i n expr : LOOP_FLAG COMPLETE_FLAG FORCE_FLAG
    break_statement ::= b r e a k NEED_LOOP_FLAG COMPLETE_FLAG
    continue_statement ::= c o n t i n u e NEED_LOOP_FLAG COMPLETE_FLAG
    return_statement ::= r e t u r n expr NEED_DEF_FLAG COMPLETE_FLAG | r e t u r n NEED_DEF_FLAG COMPLETE_FLAG
    in_args ::= in_args , expr | expr
    func_call ::= v ( in_args ) | v ( )
    expr ::= l | v | func_call | expr b expr | u expr | ( expr ) | expr a expr | Array | expr . expr | expr , expr | Dict
    """
)
now_time = time.time()
Parser(grammar, [], []).read("")
print("The time is", time.time() - now_time, "s.")
print()