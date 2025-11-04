# Primer 1: generiraj nakljucen vozel in ga reduciraj

from veckotni import nakljucen_vozel

# generiraj vozel
v = nakljucen_vozel(20)

# prikaži koordinate oglišč
print(v)

# reduciraj
v.reduciraj()

# prikaži koordinate novih oglišč
print(v)

# vizualiziraj
v.prikazi()
