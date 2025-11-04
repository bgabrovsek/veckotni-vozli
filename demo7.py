# Primer 1: generiraj torusni vozel, ga reduciraj in prikaži animacijo redukcije

from veckotni import torusni_vozel, prikazi_redukcijo, shrani_video_redukcije

p, q = 3, 2

# generiraj vozle in jih reduciraj, dokler ne dobimo vozla s 3 oglišči
v = torusni_vozel(29, p, q)

# shrani seznam vozlov, ki nastanejo tekom redukcije
zaporedje = v.zaporedje_redukcij()
print(f"Torusni vozel ({p},{q}) je bil reduciran iz", len(zaporedje[0]), "na", len(zaporedje[-1]), "vozlišč po", len(zaporedje), "korakih.")

shrani_video_redukcije(zaporedje, "redukcija-torusni.gif", fps=2)
prikazi_redukcijo(zaporedje)
