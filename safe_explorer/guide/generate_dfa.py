from ltlf2dfa.parser.ltlf import LTLfParser
from graphviz import Source

parser = LTLfParser()
formula_str = "G(y -> X z)"
formula = parser(formula_str)
print(formula)

dfa = formula.to_dfa()
with open('example.dot', 'w') as f:
    f.write(dfa)

src = Source.from_file('example.dot')
src.render(filename='example', cleanup=True)