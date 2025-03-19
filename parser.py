from dataclasses import dataclass
from typing import List, Set, Tuple, Literal, Union
# Modified based on DarkSharpness's code

ROOT_RULE= "$"
root_rule_number = 0

# The following flags are used to represent a universal symbol.
XGRAMMAR_EVERYTHING_FLAG = "XGRAMMAR_EVERYTHING_FLAG"
XGRAMMAR_DIGIT_FLAG = "XGRAMMAR_DIGIT_FLAG"
XGRAMMAR_HEX_FLAG = "XGRAMMAR_HEX_FLAG"

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
        dict = {str:int}
        cnt = 0
        results = []
        # Do the first scanning. Add all the non-terminal symbols to the dictionary.
        for line in rule.split("\n"):
            if not line.strip():
                continue
            lhs, rhs = line.replace(" ", "").split("::=")
            cnt += 1
            dict[lhs] = cnt
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
                root_rule_number = dict[lhs]
            for rule in rhs.split("|"):
                processed = []
                for symbol in rule.split(" "):
                    if(symbol == ""):
                        continue
                    if symbol in dict:
                        processed.append(dict[symbol])
                    else:
                        processed.append(symbol)    
                if(processed != []):
                    results += [(cnt, processed)]
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

    def __post_init__(self):
        self.state_set: List[Set[State]] = [set()]
        self.inputs: str = ""
        self.state_set[0].add(State(*self.grammar[root_rule_number][0]))
        
    def _complete(self, state: State) -> List[State]:
        results: List[State] = []
        for r in self.state_set[state.start]:
            if state.name == r.symbol():
                results.append(next(r))
        return results

    def _predict(self, start: int, symbol: str) -> List[State]:
        return [
            State(*r, start=start)
            for r in self.grammar[symbol]
        ]

    def _scan(self, state: State, start: int, token: str):
        if state.symbol() == token or (state.symbol() == XGRAMMAR_EVERYTHING_FLAG and state.symbol() != "\\") or (token.isdigit() and state.symbol() == XGRAMMAR_DIGIT_FLAG) or (state.symbol() == XGRAMMAR_HEX_FLAG and token in "0123456789abcdefABCDEF"):
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
            accept = any(s.name == root_rule_number and s.terminated() for s in state)
            print(f"State {i}: {text[:i]}•{text[i:]} {accept=}")
            print("\n".join(f"  {s}" for s in state))

    def _print(self, pos: int) -> None:
        copy = Parser(self.grammar)
        copy.state_set = self.state_set + []
        copy.inputs = self.inputs + ""
        return copy._finalize(pos)

    def read(self, text: str):
        length = len(self.state_set)
        for token in text:
            self._consume(token)
        self._print(length)
        return self
# In my realization, $ shouldn't have multiple rules. i.e.
# $ ::= Array | Object is undefined.
# If we want to use a rule to parse a "word", 
# we need to use whitespace to separate them.
grammar = Grammar.parse(
    """
    $ ::= Json
    Json ::= Array | Object
    Array ::= LeftBrace Element RightBrace
    Object ::= LeftBracket ObjectElement RightBracket
    LeftBrace ::= [
    RightBrace ::= ]
    LeftBracket ::= {
    RightBracket ::= }
    Quote ::= "
    Comma ::= ,
    Dot ::= .
    Colon ::= :
    ObjectElement ::= String Colon Value Comma ObjectElement | String Colon Value
    Element ::= Value Comma Element | Value
    Value ::= String | Int | Float | Object | Array | Bool | Null
    Float ::= Int Dot Int
    Int ::= XGRAMMAR_DIGIT_FLAG | Int XGRAMMAR_DIGIT_FLAG
    String ::= Quote Quote | Quote chars Quote 
    chars ::= XGRAMMAR_EVERYTHING_FLAG | chars XGRAMMAR_EVERYTHING_FLAG | chars escaped | escaped
    escaped ::= escaped_char Quote | escaped_char / | escaped_char n | escaped_char b | escaped_char f | escaped_char r | escaped_char t | escaped_char u XGRAMMAR_HEX_FLAG XGRAMMAR_HEX_FLAG XGRAMMAR_HEX_FLAG XGRAMMAR_HEX_FLAG
    escaped_char ::= \\
    Bool ::= t r u e | f a l s e
    Null ::= n u l l
    """
)

# print(Parser(grammar))
Parser(grammar).read("[{\"a\":\"\\u13DF\"},null]")