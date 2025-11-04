# Primer 1: generiraj naključne vozle in jih reduciraj toliko časa, dokler se ne najde vozel,
# ki ga ni mogoče reducirati na manj kot 6 oglišč (deteljica)

from veckotni import nakljucen_vozel, poisci_reduciran_vozel

# generiraj vozle in jih reduciraj
v = poisci_reduciran_vozel(15, 6)

# prikaži koordinate oglišč
print(v)

# vizualiziraj
v.prikazi()