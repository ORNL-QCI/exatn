grammar TAProL;

taprolsrc
   : entry (scope)+
   | code
   ;

entry
   : 'entry' ':' scopename (';')?
   ;

scope
   : 'scope' scopename 'group' '(' groupnamelist? ')' (';')? code 'end' 'scope' scopename (';')?
   ;

scopename
   : id
   ;

groupnamelist
   : groupname (',' groupname)*
   ;

groupname
   : id
   ;

code
   : (line)*
   ;

line
   : statement (';')?
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
   : assign
   | retrieve
   | load
   | save
   | destroy
   | norm1
   | norm2
   | maxabs
   | scale
   | copy
   | addition
   | contraction
   ;

compositeop
   : compositeproduct
   | tensornetwork
   ;

space
   : 'space' '(' numfield ')' ':' spacedeflist
   ;

numfield
   : 'real'
   | 'complex'
   ;

subspace
   : 'subspace' '(' (spacename)? ')' ':' spacedeflist
   ;

spacedeflist
   : spacedef (',' spacedef)*
   ;

spacedef
   : spacename '=' range
   ;

spacename
   : id
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

index
   : 'index' '(' spacename ')' ':' indexlist
   ;

indexlist
   : indexname (',' indexname)*
   ;

indexname
   : id
   ;

assign
   : tensor '=' '?'
   | tensor '=' (real | complex)
   | tensor '=' datacontainer
   | tensor '=' 'method' '(' methodname ')'
   | tensor '=>' 'method' '(' methodname ')'
   ;

datacontainer
   : id
   ;

methodname
   : string
   ;

retrieve
   : datacontainer '=' tensorname
   | datacontainer '=' tensor
   ;

load
   : 'load' (tensor | tensorname) ':' 'tag' '(' tagname ')'
   ;

save
   : 'save' (tensor | tensorname) ':' 'tag' '(' tagname ')'
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

norm1
   : scalar '=' 'norm1' '(' (tensorname | tensor) ')'
   ;

norm2
   : scalar '=' 'norm2' '(' (tensorname | tensor) ')'
   ;

maxabs
   : scalar '=' 'maxabs' '(' (tensorname | tensor) ')'
   ;

scalar
   : id
   ;

scale
   : tensor '*=' prefactor
   ;

prefactor
   : real
   | complex
   | id
   ;

copy
   : tensor '=' tensor
   ;

addition
   : tensor '+=' (tensor | conjtensor) ( '*' prefactor )?
   ;

contraction
   : tensor '+=' (tensor | conjtensor) '*' (tensor | conjtensor) ( '*' prefactor )?
   ;

compositeproduct
   : tensor '+=' (tensor | conjtensor) '*' (tensor | conjtensor) ( '*' (tensor | conjtensor) )+ ( '*' prefactor )?
   ;

tensornetwork
   : tensor '=>' (tensor | conjtensor) ( '*' (tensor | conjtensor) )+
   ;

tensor
   : tensorname '(' (indexlist)? ')'
   ;

conjtensor
   : tensorname '+' '(' (indexlist)? ')'
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
   | FLOAT
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

/* Fixed point real number */
REAL
   : ('-')? INT '.' ZINT
   ;

/* Floating point real number */
FLOAT
   : ('-')? INT 'e' ('+' | '-')? INT
   | REAL 'e' ('+' | '-')? INT
   ;

/* Non-negative integer */
INT
   : '0'
   | ('1'..'9') ('0'..'9')*
   ;

/* Non-negative integer with posible leading zeros */
ZINT
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
