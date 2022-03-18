# This demonstrates that there are many strings which can
# by resolved to the canonical reaction names.
from fusrate.reactionnames import name_resolver

def resolve(s):
    print(s + ' is ' + name_resolver(s))

print("Deuterium-tritium fusion canonical reaction name: "
        + name_resolver('DT'))
resolve('DT')
resolve('DT')
resolve('D-T')
resolve('D+T')
resolve('D+T→n+α')
resolve('D+T→α+n')

print('')
print("Deuterium-helium fusion canonical reaction name: "
        + name_resolver('DHe3'))
resolve('DHe3')
resolve('DHe')
resolve('D3He')
resolve('D+3He')
resolve('D+³He')
resolve('D+³He→p+⁴He')
resolve('D+³He→p+α')
resolve('D+³He→α+p')
resolve('D+³He->a+p')

print('')
print("Deuterium-deuterium tritium-producing "
"reaction canonical names: " + name_resolver('D+D→p+T'))
resolve('D+D→p+T')
resolve('D+D→T+p')
resolve('²H+²H→³H+¹H')
resolve('²H+²H→¹H+³H')

print('')
print("Deuterium-deuterium helion-producing "
"reaction canonical names: " + name_resolver('D(d,n)3He'))
resolve('D+D→n+3He')
resolve('D+D→3He+n')
resolve('²H+²H→n+3He')
resolve('²H+²H→3He+n')

print('')
print('Proton-boron fusion: ' + name_resolver('pB'))
resolve('pB')
resolve('pB11')
resolve('p+B')
resolve('p+B11')
resolve('p+11B')
resolve('p+11B→3α')
resolve('p+11B→3 ⁴He')
