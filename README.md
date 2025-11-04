# \# Knjižnica za večkotne vozle

# 

# Knjižnica za večkotne (poligonalne) vozle v tridimenzionalnem prostoru.

# 

# ---

# 

# \## Opis

# 

# Večkotni vozel je definiran kot zaporedje točk v tridimenzionalnem prostoru, podanih v obliki:

# 

# (x, y, z)

# 

# Zaporedne točke so povezane z daljicami, pri čemer zadnja točka ponovno poveže prvo, tako da nastane zaprt poligon.

# 

# Knjižnica omogoča:

# \- preverjanje pravilnosti strukture vozla (brez kolinearnih zaporednih točk in brez samopresečišč),

# \- vizualizacijo v 3D prostoru,

# \- izvoz podatkov v obliko, primerno za GeoGebro,

# \- aproksimacijo torusnih vozlov s poligonalnimi točkami,

# \- redukcijo vozla z odstranjevanjem odvečnih oglišč.

# 

# ---

# 

# \## Namestitev

# 

# Za delovanje knjižnice zadostuje Python 3.9 ali novejši ter knjižnici `numpy` in `matplotlib`:

# 

# ```bash

# pip install numpy matplotlib



