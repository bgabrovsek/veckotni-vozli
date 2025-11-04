# Primer 1: generiraj nakljucni vozel, ga reduciraj in prikaži animacijo redukcije

from veckotni import nakljucen_vozel, prikazi_redukcijo, shrani_video_redukcije

# generiraj vozle in jih reduciraj, dokler ne dobimo vozla s 3 oglišči
while True:
    v = nakljucen_vozel(30)

    # shrani seznam vozlov, ki nastanejo tekom redukcije
    zaporedje = v.zaporedje_redukcij()

    if len(zaporedje[-1]) == 3:
        break

print("Večkotni vozel je bil reduciran na", len(zaporedje[-1]), "vozlišč po", len(zaporedje), "korakih.")

shrani_video_redukcije(zaporedje, "redukcija-nakljucen.gif", fps=2)
prikazi_redukcijo(zaporedje)
