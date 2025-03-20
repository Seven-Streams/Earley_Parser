grammar peg;
grammars:
   matcher*;
keyword:
   '\'' KEYWORDNAME '\'';
soft_keyword:
   '"' KEYWORDNAME '"';
matcher:
EXPRESSIONWORD ':' rules
;
rules:
   rules '|' expression |
   expression
;
expression:
   expression expression #apend
   | expression '|' expression #or
   | keyword  #key
   | soft_keyword #s_key
   | EXPRESSIONWORD #match
   | '(' expression ')' #s_bracket
   | '[' expression ']' #m_bracket
   | expression '?' #question
   | expression '*' #star
   | keyword '.' expression '+' #separate
   | '&' expression # success_no_consume
   | '!' expression # fail_no_consume
   | '~' expression # commit
   
;
KEYWORDNAME: [!-~]+;
EXPRESSIONWORD: [a-zA-Z0-9_];
COMMENT: '#' ~[\r\n]* -> skip;
WS: [ \r\n\t]+ -> skip;