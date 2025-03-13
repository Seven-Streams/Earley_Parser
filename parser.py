from dataclasses import dataclass
from typing import List, Set, Tuple, Literal, Union
# Modified based on DarkSharpness's code

ROOT_RULE= 0
ROOT_RULE_NUMBER = 0
# Terminal = Literal[
#     "a", "b", "c", "d", "e", "f", "g", "h", "i",
#     "j", "k", "l", "m", "n", "o", "p", "q", "r",
#     "s", "t", "u", "v", "w", "x", "y", "z", ":",
#     "{", "}", "[", "]", "\"" 
# ]
# NonTerminal = Literal[
#     "A", "B", "C", "D", "E", "F", "G", "H", "I",
#     "J", "K", "L", "M", "N", "O", "P", "Q", "R",
#     "S", "T", "U", "V", "W", "X", "Y", "Z", "$"
# ]



# def is_terminal(symbol: str) -> Terminal | None:
#     return symbol if symbol.islower() else None # type: ignore

# def is_nonterminal(symbol: str) -> NonTerminal | None:
#     if symbol.isupper() or symbol == ROOT_RULE:
#         return symbol # type: ignore
#     return None

# # Upper case -> non-terminal
# # Lower case -> terminal

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
                ROOT_RULE_NUMBER = dict[lhs]
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
    expr: str
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

END_SYMBOL = "."

@dataclass
class Parser:
    grammar: Grammar

    def __post_init__(self):
        self.state_set: List[Set[State]] = [set()]
        self.inputs: str = ""
        # self.state_set[0].add(State(*self.grammar[ROOT_RULE_NUMBER][0]))
        
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
        if state.symbol() == token:
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
            accept = any(s.name == ROOT_RULE and s.terminated() for s in state)
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

grammar = Grammar.parse(
    """
    $ ::= A
    A ::= a A B | b | c
    B ::= A | a |
    """
)

print(Parser(grammar))
# Parser(grammar).read("ab")