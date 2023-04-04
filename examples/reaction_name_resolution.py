# This demonstrates that there are many strings which can
# by resolved to the canonical reaction names.
from fusionrate.reactionnames import name_resolver


def resolve(s):
    print(s + " is " + name_resolver(s))


print(
    "Deuterium-tritium fusion canonical reaction name: " + name_resolver("DT")
)
resolve("DT")
resolve("D+T")
resolve("D+T→n+α")
resolve("D+T→α+n")
resolve("t(d,n)a")

print("")
print(
    "Deuterium-helium fusion canonical reaction name: " + name_resolver("DHe3")
)
resolve("DHe3")
resolve("DHe")
resolve("D3He")
resolve("D+3He")
resolve("D+³He")
resolve("D+³He→p+⁴He")
resolve("D+³He→p+α")
resolve("D+³He→α+p")
resolve("D+³He->a+p")
resolve("h(d,p)a")

print("")
print(
    "Deuterium-deuterium tritium-producing "
    "reaction canonical name: " + name_resolver("D+D→p+T")
)
resolve("D+D→p+T")
resolve("D+D→T+p")
resolve("²H+²H→³H+¹H")
resolve("²H+²H→¹H+³H")
resolve("d(d,p)t")

print("")
print(
    "Deuterium-deuterium helion-producing "
    "reaction canonical name: " + name_resolver("D(d,n)3He")
)
resolve("D+D→n+3He")
resolve("D+D→3He+n")
resolve("²H+²H→n+3He")
resolve("²H+²H→3He+n")
resolve("d(d,n)h")

print("")
print("Tritium-tritium fusion canonical reaction name: " + name_resolver("T+T"))
resolve("2T")
resolve("T+T")
resolve("T + T -> a + 2n")
resolve("t(t,2n)a")

print("")
print("Proton-boron fusion canonical reaction name: " + name_resolver("pB"))
resolve("pB")
resolve("pB11")
resolve("p+B")
resolve("p+11B")
resolve("p+11B→3α")
resolve("p+11B→3 ⁴He")

print("")
print("Tritium-helion proton-and-neutron-producing "
      "reaction canonical name: " + name_resolver("³He(t,pn)⁴He"))
resolve("h + t -> p + n + a")
resolve("h(t,pn)a")
resolve("h(t,np)a")

print("")
print("Tritium-helion deuterium-producing "
      "reaction canonical name: " + name_resolver("³He(t,d)⁴He"))
resolve("h + t -> d + a")
resolve("h(t,d)a")

print("")
print("Helion-helion "
      "reaction canonical name: " + name_resolver("³He(h,2p)⁴He"))
resolve("h + h -> 2 p + a")
resolve("h(h,2p)a")

print("")
print("Proton-lithium fusion canonical reaction name: " + name_resolver("pLi6"))
resolve("pLi6")
resolve("p + ⁶Li")
resolve("p + ⁶Li --> h + α")
resolve("6Li(p,h)a")

print("")
print("Deuteron-lithium-6 alpha-producing "
      "reaction canonical name: " + name_resolver("6Li(d,a)a"))
resolve("6Li(d,a)a")

print("")
print("Deuteron-lithium-6 Beryllium-producing "
      "reaction canonical name: " + name_resolver("6Li(d,n)Be"))
resolve("6Li(d,n)Be")

print("")
print("Deuteron-lithium-6 Lithium-7-producing "
      "reaction canonical name: " + name_resolver("6Li(d,p)7Li"))
resolve("6Li(d,p)7Li")
resolve("6Li + d --> p + 7Li")
