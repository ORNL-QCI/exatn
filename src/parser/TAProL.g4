grammar TAProL;

taprolsrc
   : entry (scope)+
   ;

entry
   : 'entry' ':' entryname
   ;

entryname
   : id
   ;

scope
   : 'scope' scopename 'group' '(' groupnamelist? ')' code 'end'
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
   : (line)+
   ;

line
   : statement
   | comment
   ;

statement
   : space EOL
   | subspace EOL
   | index EOL
   | simpleop EOL
   | compositeop EOL
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
   : 'index' '(' subspacename ')' ':' indexlist
   ;

subspace
   : 'subspace' '(' (spacename)? ')' ':' subspacelist
   ;

subspacelist
   : subspacename '=' range (',' subspacelist)*
   ;

subspacename
   : id
   ;

space
   : 'space' '(' ('real' | 'complex') ')' ':' spacelist
   ;

spacelist
   : spacename '=' range (',' spacelist)*
   ;

spacename
   : id
   ;

assignment
   : tensor '=' '?'
   | tensor '=' (real | complex)
   | tensor '=' 'method' '(' string ')'
   | tensor '=>' 'method' '(' string ')'
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
   | 'destroy' (tensorname | tensor) (',' (tensorname | tensor) )*
   ;

copy
   : tensor '=' tensor
   ;

scale
   : tensor '*=' (real | complex | id)
   ;

unaryop
   : tensor '+=' (tensor | conjtensor) ( '*' (real | complex | id) )?
   ;

binaryop
   : tensor '+=' (tensor | conjtensor) '*' (tensor | conjtensor) ( '*' (real | complex | id) )?
   ;

compositeproduct
   : tensor '+=' (tensor | conjtensor) '*' (tensor | conjtensor) ( '*' (tensor | conjtensor) )+ ( '*' (real | complex | id) )?
   ;

tensornetwork
   : tensor '=>' (tensor | conjtensor) ( '*' (tensor | conjtensor) )+
   ;

conjtensor
   : tensorname '+' '(' (indexlist)? ')'
   ;

tensor
   : tensorname '(' (indexlist)? ')'
   ;

tensorname
   : id
   ;

tensormodelist
   : tensormode (',' tensormode)*
   ;

tensormode
   : indexlabel
   | range
   ;

indexlist
   : indexlabel (',' indexlabel)*
   ;

indexlabel
   : id
   ;

range
   : '[' (INT | id) ':' (INT | id) ']'
   ;

/* TAProL identifier */
id
   : ID
   ;

complex
   : '{' real ',' real '}'
   ;

real
   : REAL
   ;

/* Strings are enclosed in quotes */
string
   : STRING
   ;

/* A program comment */
comment
   : COMMENT
   ;

/* Tokens for the grammar */

/* Comment */
COMMENT
   : '#' ~ [\r\n]* EOL
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
