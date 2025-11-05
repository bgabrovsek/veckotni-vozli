Knjižnica za večkotne vozle
===========================

Knjižnica za večkotne (poligonalne) vozle v tridimenzionalnem prostoru.

Domonstracija knjižnice je na voljo na povezavi
https://bgabrovsek.github.io/veckotni-vozli/lab/index.html

------------------------------------------------------------

Opis
-----

Večkotni vozel (ang. polygonal knot) je topološki objekt, predstavljen z zaporedjem točk v tridimenzionalnem prostoru. Točke so povezane z daljicami, pri čemer zadnja točka ponovno poveže prvo, tako da dobimo zaprt poligon. Takšna predstavitev omogoča numerično, geometrijsko in topološko obdelavo vozlov, kar je uporabno pri računski topologiji, geometrijskem modeliranju in analizi proteinskih struktur.

Knjižnica omogoča:
- preverjanje pravilnosti strukture vozla (brez kolinearnih zaporednih točk in brez samopresečišč),
- vizualizacijo večkotnega vozla v 3D prostoru,
- izvoz podatkov v obliko, primerno za uporabo v programu GeoGebra,
- aproksimacijo torusnih vozlov s poligonalnimi točkami,
- postopno redukcijo vozla z odstranjevanjem odvečnih oglišč,
- vizualno animacijo postopka redukcije v obliki GIF datoteke ali interaktivnega prikaza z drsnikom.

------------------------------------------------------------

Teoretično ozadje
------------------

Večkotni (poligonalni) vozli so diskretna aproksimacija gladkih vozlov, ki se pojavljajo v topologiji in fiziki. Formalno je vozel vložek krožnice S¹ v evklidski prostor R³. V računalniški praksi tak vložek predstavimo z zaporedjem točk (x_i, y_i, z_i), kjer so zaporedne točke povezane z ravnimi daljicami.
Redukcija večkotnega vozla pomeni postopno odstranjevanje oglišč, pri čemer se ohrani njegova topološka struktura. To omogoča učinkovitejše prikaze in hitrejše izračune, ne da bi se spremenil tip vozla.

------------------------------------------------------------

Namestitev
-----------

Knjižnica je napisana v jeziku Python in uporablja le standardne znanstvene knjižnice.

Za delovanje zadostuje:
- Python 3.9 ali novejši
- knjižnici numpy in matplotlib

Namestitev zahtevanih paketov:

pip install numpy matplotlib

------------------------------------------------------------

Uporaba
--------

Primer 1: ustvarjanje in prikaz večkotnega vozla

from vozel import Vozel
import numpy as np

tocke = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [2, 0, 0],
    [1, -1, 0]
])

v = Vozel(tocke)
v.prikazi()

Primer 2: redukcija vozla in prikaz postopka

koraki = v.reduciraj_z_zgodovino()

from vizualizacija import shrani_video_redukcije
shrani_video_redukcije(koraki, "redukcija.gif", fps=2)

------------------------------------------------------------

Primer generiranja torusnega vozla
----------------------------------

from torus import torus_vozel

v = torus_vozel(p=3, q=2, n=200)
v.prikazi()

------------------------------------------------------------

Avtorji
-------------------------------
Eva Horvat  
Boštjan Gabrovšek  
Luka Demšar  
Ivo Štukelj
Teja Sajovic  

