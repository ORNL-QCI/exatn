grammar TAProL;

taprolsrc
   : entry
   | scope+
   ;

entry
   : 'entry:' ID
   ;

scope
   : 'scope' scopename=id 'group' '(' groupnamelist? ')' code
   ;

code
   : (line)+
   ;

line
   : statement+
   | comment
   ;

/*
entry: main # First function to start from, hence ENTRY
scope main group() # Function definition
 subspace(): s0=[0:127]
 index(s0): a,b,c,d,i,j,k,l
 H2(a,b,c,d) = method("HamiltonianTest")
 T2(a,b,c,d) = {1.0,0.0}
 Z2(a,b,c,d) = {0.0,0.0}
 Z2(a,b,c,d) += H2(i,j,k,l) * T2(c,d,i,j) * T2(a,b,k,l)
 X2() = {0.0,0.0}
 X2() += Z2+(a,b,c,d) * Z2(a,b,c,d)
 save X2: tag("Z2_norm")
 ~X2
 ~Z2
 ~T2
 ~H2
end scope main
*/

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

index
   : 'index' '(' subspacename=id ')' ':' id (',' id)?
   ;

assignment
   : tensor '=' '?'
   | tensor '=' 'method' '(' id ')'
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
   ;

conjtensor
   : tensorname '+' '(' (indexlist)? ')'
   ;

actualindex
   : id
   | INT
   ;

indexlist
   : actualindex (',' actualindex)?
   ;

/* A program comment */
comment
   : COMMENT
   ;

spacelist
   : spacename=id '=' range (',' spacelist)?
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
