# Primer 1: generiraj torusni vozel, ga reduciraj in prikaži

from veckotni import torusni_vozel

# generiraj vozle in jih reduciraj
v = torusni_vozel(29, 3, 2)

# prikaži koordinate oglišč
print(v)

v.reduciraj()

# prikaži koordinate oglišč
print(v)

# vizualiziraj
v.prikazi()