grammar TAProL;

taprolsrc
   : entry (scope)+
   | code
   ;

entry
   : 'entry' ':' entryname=id
   ;

scope
   : 'scope' scopename=id 'group' '(' groupnamelist? ')' code 'end scope' endscopename=id 
   ;

groupnamelist
   : groupname=id (',' groupname=id)*
   ;

code
   : (line)+
   ;

line
   : statement
   | comment
   ;

statement
   : space 
   | subspace 
   | index 
   | simpleop 
   | compositeop 
   ;

compositeop
   : compositeproduct
   | tensornetwork
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

index
   : 'index' '(' spacename ')' ':' indexlist
   ;

indexlist
   : indexname (',' indexname)*
   ;

indexname
   : id
   ;

subspace
   : 'subspace' '(' (spacename)? ')' ':' spacedeflist
   ;

space
   : 'space' '(' numfield ')' ':' spacedeflist
   ;

spacename
   : id
   ;

numfield
   : 'real'
   | 'complex'
   ;

spacedeflist
   : spacedef (',' spacedef)*
   ;

spacedef
   : spacename '=' range
   ;

range
   : '[' lowerbound ':' upperbound ']'
   ;

lowerbound
   : INT
   | id
   ;

upperbound
   : INT
   | id
   ;

assignment
   : tensor '=' '?'
   | tensor '=' (real | complex)
   | tensor '=' 'method' '(' methodname ')'
   | tensor '=>' 'method' '(' methodname ')'
   ;

methodname
   : string
   ;

load
   : 'load' tensor ':' 'tag' '(' tagname ')'
   ;

save
   : 'save' tensor ':' 'tag' '(' tagname ')'
   ;

tagname
   : string
   ;

destroy
   : '~' tensorname
   | '~' tensor
   | 'destroy' tensorlist
   ;

tensorlist
   : (tensorname | tensor) (',' (tensorname | tensor) )*
   ;

copy
   : tensor '=' tensor
   ;

scale
   : tensor '*=' prefactor
   ;

unaryop
   : tensor '+=' (tensor | conjtensor) ( '*' prefactor )?
   ;

binaryop
   : tensor '+=' (tensor | conjtensor) '*' (tensor | conjtensor) ( '*' prefactor )?
   ;

prefactor
   : real
   | complex
   | id
   ;

compositeproduct
   : tensor '+=' (tensor | conjtensor) '*' (tensor | conjtensor) ( '*' (tensor | conjtensor) )+ ( '*' prefactor )?
   ;

tensornetwork
   : tensor '=>' (tensor | conjtensor) ( '*' (tensor | conjtensor) )+
   ;

conjtensor
   : tensorname '+' '(' (indexlist)? ')'
   ;

tensor
   : tensorname ('(' (indexlist)? ')')?
   ;

tensorname
   : id
   ;

id
   : ID
   ;

complex
   : '{' real ',' real '}'
   ;

real
   : REAL
   ;

string
   : STRING
   ;

comment
   : COMMENT
   ;

/* Tokens for the grammar */

/* Comment */
COMMENT
   : '#' ~ [\r\n]* 
   ;

/* Alphanumeric_ identifier */
ID
   : [A-Za-z][A-Za-z0-9_]*
   ;

/* Real number */
REAL
   : ('-')? INT '.' INT
   ;

/* Non-negative integer */
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
