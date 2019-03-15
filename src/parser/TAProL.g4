grammar TAProL;

taprolsrc
   : entry (scope)+
   ;

entry
   : 'entry:' entryName=ID
   ;

scope
   : 'scope' scopebeginname=id 'group' '(' groupnamelist? ')' code 'end' 'scope' scopeendname=id
   ;

code
   : (line)+
   ;

line
   : statement+
   | comment
   ;

statement
   : space
   | subspace
   | index
   | simpleop
   | compositeop
   ;

simpleop
   : assignment
   | load
   | save
   | destroy
   | copy
   | scale
   | unaryop
   | binaryop
   ;

compositeop
   : compositeproduct
   | tensornetwork
   ;

space
   : 'space' '(' ('real'|'complex') ')' ':' spacelist
   ;

subspace
   : 'subspace' '(' (spacename=id)? ')' ':' spacelist
   ;

spacelist
   : spacename=id '=' range (',' spacelist)+?
   ;

index
   : 'index' '(' subspacename=id ')' ':' idx (',' idx)+?
   ;

idx
   : id
   ;

assignment
   : tensor '=' '?'
   | tensor '=' 'method' '(' string ')'
   | tensor '=' complex
   ;

load
   : 'load' tensor ':' 'tag' '(' string ')'
   ;

save
   : 'save' tensor ':' 'tag' '(' string ')'
   ;

destroy
   : '~' tensorname
   | '~' tensor
   | 'destroy' (tensorname | tensor) (',' (tensorname | tensor) )?
   ;

copy
   : tensor '=' tensor
   ;

scale
   : tensor '*=' (real | complex | id)
   ;

unaryop
   : tensor '+=' (tensor | conjtensor) ( '*' (real | complex | id))?
   ;

binaryop
   : tensor '+=' (tensor | conjtensor) '*' (tensor | conjtensor) ( '*' (real | complex | id) )?
   ;

compositeproduct
   : tensor '+=' (tensor | conjtensor) '*' (tensor | conjtensor) ( '*' (tensor|conjtensor) )+ ( '*' (real | complex | id) )?
   ;

tensornetwork
   : tensor '=>' (tensor | conjtensor) ( '*' (tensor|conjtensor) )+
   ;

tensorname
   : id
   ;

tensor
   : tensorname '(' (indexlist)? ')'
   | tensorname
   ;

conjtensor
   : tensorname '+' '(' (indexlist)? ')'
   ;

actualindex
   : id
   | INT
   ;

indexlist
   : actualindex (',' actualindex)+?
   ;

/* A program comment */
comment
   : COMMENT
   ;

range
   : '[' (INT | id) ':' (INT | id) ']'
   ;

groupnamelist
   : groupname (',' groupnamelist)?
   ;

/* A parameter */
groupname
   : id
   ;

/* variable identity */
id
   : ID
   ;

complex
   : '{' real ',' real '}'
   ;

real
   : REAL
   ;

/* strings are enclosed in quotes */
string
   : STRING
   ;

/* Tokens for the grammer */

/* Comment */
COMMENT
   : '#' ~ [\r\n]* EOL
   ;

/* id, ego, and super-ego */
ID
   : [a-z][A-Za-z0-9_]*
   | [A-Z][A-Za-z0-9_]*
   | [A-Z][A-Za-z]*
   ;

/* Keep it real...numbers */
REAL
   : INT ( '.' (INT) )
   ;

/* Non-negative integers */
INT
   : ('0'..'9')+
   ;

/* Strings include numbers and slashes */
STRING
   : '"' ~ ["]* '"'
   ;

/* Whitespaces, we skip'em */
WS
   : [ \t\r\n] -> skip
   ;

/* This is the end of the line, boys */
EOL
: '\r'? '\n'
;
