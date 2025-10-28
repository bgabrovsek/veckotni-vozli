"""
Knjižnica za večkotne (poligonalne) vozle.

Večkotni vozel je definiran kot zaporedje točk v tridimenzionalnem prostoru,
podanih v obliki (x, y, z). Zaporedne točke so povezane z daljicami, pri čemer
zadnja točka ponovno poveže prvo, tako da nastane zaprt poligon.

Knjižnica omogoča:
- preverjanje pravilnosti strukture vozla (brez kolinearnih zaporednih točk in brez samopresečišč),
- vizualizacijo v 3D prostoru,
- izvoz podatkov v obliko, primerno za GeoGebro,
- aproksimacijo torusnih vozlov s poligonalnimi točkami,
- redukcijo vozla z odstranjevanjem odvečnih oglišč.

Avtorji: Eva Horvat, Boštjan Gabrovšek, Luka Demšar, Teja Sajovic, Ivo Štukelj
Licenca: MIT
Različica: 1.0
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt



class Vozel:
    """Večkotni (poligonalni) vozel podan z zaporedjem točk v 3D prostoru.

    Razred predstavlja zaprt poligonalni niz točk (ogliščen poligon) v R^3.
    Zaporedne točke so povezane z daljicami, zadnja točka pa je povezana s prvo.

    Atributi:
        vertices (np.ndarray): Tabela oblike (N, 3), kjer vsaka vrstica
            predstavlja oglišče z koordinatami (x, y, z).
        name (Optional[str]): Poljubno ime vozla.

    Izjeme:
        ValueError: Če vhodne točke niso oblike (N, 3), če zaporedna trojica
            točk vsebuje kolinearne točke ali če ima vozel samopresečišča.
    """

    def __init__(self, vertices, name: Optional[str] = None):
        """Inicializira večkotni vozel.

        Args:
            vertices: Poljubna struktura, ki jo je mogoče pretvoriti v
                `np.ndarray` oblike (N, 3) z oglišči vozla.
            name: Ime vozla (neobvezno).
        """
        vertices = np.array(vertices)  # Pretvorba v NumPy tabelo
        self.vertices: np.ndarray = np.array(vertices)
        self.name: Optional[str] = name

        # Preverjanje oblike podatkov
        if len(self.vertices.shape) != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Oglišča morajo biti v tabeli oblike (N, 3).")

        # Preverjanje, da nobene tri zaporedne točke niso kolinearne
        ext = np.concatenate((vertices, vertices[:2]), axis=0)
        for a, b, c in zip(ext, ext[1:], ext[2:]):
            u, v = c - b, b - a
            rez = np.cross(u, v)
            if not np.all(rez):
                raise ValueError(f"Točke {a}, {b}, {c} so kolinearne.")

        # Preverjanje samopresečišč
        if not ima_samopresecisca(vertices):
            raise ValueError("Vozel ima samopresečišča.")

    def __repr__(self) -> str:
        """Vrne opisno predstavitev objekta.

        Returns:
            str: Formatiran niz z imenom (če je dano) in številom oglišč.
        """
        rezultat = "Večkotni vozel "
        rezultat += f"\"{self.name}\" " if self.name else ""
        rezultat += f"s {len(self.vertices)} oglišči:\n"
        rezultat += f"{self.vertices}"
        return rezultat

    def visualize(self) -> None:
        """Prikaže vozel v 3D prostoru z oglišči in povezavami."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        ax.scatter(x, y, z, color='blue', s=50, label="Oglišča")

        color_map = plt.cm.viridis
        n = len(self.vertices)

        # Povezave med zaporednimi oglišči
        for i in range(n - 1):
            xi, yi, zi = self.vertices[i]
            xj, yj, zj = self.vertices[i + 1]
            color = color_map(i / (n - 1))
            ax.plot([xi, xj], [yi, yj], [zi, zj], color=color, linewidth=2)

        # Zapremo poligon: zadnja -> prva
        xi, yi, zi = self.vertices[0]
        xj, yj, zj = self.vertices[-1]
        ax.plot([xi, xj], [yi, yj], [zi, zj], color=color_map(1), linewidth=2)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.legend()
        plt.show()

    def geogebra(self) -> None:
        """Izpiše niza ukazov za GeoGebro, ki izrišeta vozel kot zaporedje daljic."""
        tocke_niz = "Točke={"
        tocke_niz += ", ".join([f"({t[0]}, {t[1]}, {t[2]})" for t in self.vertices])
        tocke_niz += f", ({self.vertices[0][0]}, {self.vertices[0][1]}, {self.vertices[0][2]})}}"

        daljice_niz = "Zaporedje(Daljica(Element(Točke,i),Element(Točke,i+1)),i,1,Dolžina(Točke)-1)"

        print("Prvi niz:", tocke_niz)
        print("Drugi niz:", daljice_niz)

    def approx_torus_knot(self, p: int, q: int, vertices: int = 50) -> None:
        """Aproksimira torusni vozel kot večkotni vozel in zamenja trenutna oglišča.

        Opomba: Metoda ohrani obstoječe ime vozla, vendar oglišča
        zamenja z vzorčenimi točkami torusnega vozla.

        Args:
            p (int): Parameter torusnega vozla (skupaj z `q` naj bosta tuji).
            q (int): Parameter torusnega vozla (skupaj z `p` naj bosta tuji).
            vertices (int): Število oglišč aproksimacije.
        """
        interval = np.linspace(0, 2 * math.pi, vertices, endpoint=False)
        self.vertices = np.array(
            [
                [
                    (math.cos(t * q) + 2) * math.cos(p * t),
                    (math.cos(t * q) + 2) * math.sin(p * t),
                    -1 * math.sin(q * t),
                ]
                for t in interval
            ]
        )

    def reduciraj(self) -> int:
        """Popolnoma reducira večkotni vozel z odstranjevanjem odvečnih oglišč.

        Algoritem iterativno preverja trojke zaporednih točk in,
        kadar je to dovoljeno (podano z `lahko_odstranim_ogljisce`),
        odstrani sredinsko oglišče. Postopek se konča, ko ni več
        mogoče odstraniti nobene točke.

        Returns:
            int: Število odstranjenih oglišč.
        """
        odstranjenih = 0
        dolzina = len(self.vertices)

        if dolzina == 3:
            return 0

        while True:
            if dolzina == 4:
                self.vertices = self.vertices[0:3]
                return odstranjenih + 1

            for i in range(dolzina):
                # p0, p1, p2 so oglišča trikotnika; preverjamo, ali lahko izločimo p1.
                p0 = self.vertices[i % dolzina]
                p1 = self.vertices[(i + 1) % dolzina]
                p2 = self.vertices[(i + 2) % dolzina]

                # Če ni težav, odstranimo sredinsko oglišče.
                if lahko_odstranim_tocko(p0, p1, p2, self.vertices, i):
                    self.vertices = np.delete(self.vertices, (i + 1) % dolzina, 0)
                    dolzina -= 1
                    odstranjenih += 1
                    # Po vsaki odstranitvi ponovno preverimo posodobljen vozel.
                    break
            else:
                # Ni več mogoče odstraniti nobenega oglišča.
                return odstranjenih



def ima_samopresecisca(tocke: np.ndarray) -> bool:
    """Preveri, ali ima večkotni vozel samopresečišča.

    Funkcija preveri, ali se v podanem zaporedju točk (torej v nizu
    daljic, ki jih povezujejo) katerikoli dve daljici sekata drugje
    kot v skupnem oglišču. Če takega presečišča ni, funkcija vrne True.

    Args:
        tocke (np.ndarray): Zaporedje točk oblike (N, 3).

    Returns:
        bool: True, če se nobeni dve daljici ne sekata razen v skupnih ogliščih;
              False, če vozel vsebuje samopresečišče.
    """

    tocke = np.array(tocke)

    def mejno_presecisce(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> bool:
        """Preveri, ali se nosilki daljic AB in CD sekata zunaj notranjosti daljic."""
        i = 0
        alfa, beta = 0, 0
        while True:
            imenovalec_beta = (D[i] - C[i]) * (B[(i + 1) % 3] - A[(i + 1) % 3]) + (
                C[(i + 1) % 3] - D[(i + 1) % 3]
            ) * (B[i] - A[i])
            imenovalec_alfa = B[i] - A[i]

            # Če je premica "vertikalna", preskoči na naslednjo koordinato
            if imenovalec_beta == 0 or imenovalec_alfa == 0:
                i += 1
                continue

            beta = ((C[(i + 1) % 3] - A[(i + 1) % 3]) * (B[i] - A[i]) + (A[i] - C[i]) *
                    (B[(i + 1) % 3] - A[(i + 1) % 3])) / imenovalec_beta
            alfa = (C[i] - A[i] + beta * (D[i] - C[i])) / imenovalec_alfa
            break

        # Vrne True, če se sekata samo na robu (ne v notranjosti)
        return not (0 < alfa < 1 and 0 < beta < 1)

    n = len(tocke)
    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            A, B = tocke[i], tocke[i + 1]
            C, D = tocke[j], tocke[j + 1]

            v1 = B - A
            v2 = D - C

            # Če (v1 × v2) ⋅ (A − C) ≠ 0, se nosilki daljic ne sekata
            if np.dot(np.cross(v1, v2), (A - C)) != 0:
                continue

            # Do tukaj pridejo le daljice, katerih nosilki se sekata
            if mejno_presecisce(A, B, C, D):
                continue

            # Najdeno samopresečišče
            return False

    return True


def lahko_odstranim_tocko(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                          tocke: np.ndarray, i: int) -> bool:
    """Preveri, ali lahko točko p1 odstranimo iz vozla, ne da bi nastalo presečišče.

    Funkcija preveri, ali bi odstranitev točke p1 (in s tem združitev daljic p0–p1 in p1–p2
    v eno samo daljico p0–p2) povzročila presečišče nove daljice s katerim koli
    drugim delom vozla.

    Natančnost računanja je prilagojena za koordinate s približno 10 decimalnimi mesti.

    Args:
        p0 (np.ndarray): Prva točka trikotnika.
        p1 (np.ndarray): Točka, za katero preverjamo, ali jo lahko odstranimo.
        p2 (np.ndarray): Tretja točka trikotnika.
        tocke (np.ndarray): Celoten seznam oglišč vozla (oblike (N, 3)).
        i (int): Indeks točke p0 v seznamu `tocke`.

    Returns:
        bool: True, če točko p1 lahko odstranimo brez presečišča; False sicer.

    Glej tudi:
        - Line–plane intersection: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    """
    # Vektorja trikotnika
    p10 = p0 - p1
    p12 = p2 - p1
    n = np.cross(p10, p12)  # Normala na ravnino trikotnika

    dolzina = len(tocke)

    for l in range(i + 2, i + dolzina):
        # krajišči daljice in njen smerni vektor
        la = tocke[l % dolzina]
        lb = tocke[(l + 1) % dolzina]
        lab = lb - la

        # Skalarni produkt normale trikotnika in smernega vektorja daljice
        produkt_n_lab = np.dot(n, -lab)

        # 1. primer: ravnina in daljica sta vzporedni
        if round(produkt_n_lab, 10) == 0:
            # Preverimo, ali leži premica na isti ravnini
            if round(np.dot((la - p0), n), 10) == 0:
                # Preveri, ali kakšna točka daljice leži v trikotniku
                d00 = np.dot(p10, p10)
                d01 = np.dot(p10, p12)
                d11 = np.dot(p12, p12)
                imenovalec = d00 * d11 - d01 * d01

                for lt in [la, lb]:
                    # Če je točka ravno eno od oglišč trikotnika, jo preskočimo
                    if np.allclose(lt, p0) or np.allclose(lt, p2):
                        continue

                    p0t = lt - p1
                    d10 = np.dot(p0t, p10)
                    d12 = np.dot(p0t, p12)

                    lambda1 = (d11 * d10 - d01 * d12) / imenovalec
                    lambda2 = (d00 * d12 - d01 * d10) / imenovalec
                    lambda3 = 1 - lambda1 - lambda2

                    # Če točka leži znotraj trikotnika, odstranitev ni dovoljena
                    if (
                        0 <= round(lambda1, 10) <= 1
                        and 0 <= round(lambda2, 10) <= 1
                        and 0 <= round(lambda3, 10) <= 1
                    ):
                        return False
            # Ne združuj pogojev – če premica ne leži na ravnini, a je vzporedna, preskoči
            continue

        # 2. primer: ravnina in daljica nista vzporedni → preverimo presečišče
        t = np.dot(np.cross(p10, p12), la - p1) / produkt_n_lab
        t = round(t, 10)

        if 0 <= t <= 1:  # daljica seka ravnino trikotnika
            if (t == 0 and (np.allclose(la, p0) or np.allclose(la, p2))) or (
                t == 1 and (np.allclose(lb, p0) or np.allclose(lb, p2))
            ):
                continue

            # Koordinate presečišča z ravnino v baricentrični obliki
            u = np.dot(np.cross(p12, -lab), (la - p1)) / produkt_n_lab
            v = np.dot(np.cross(-lab, p10), (la - p1)) / produkt_n_lab

            # Če daljica seka notranjost trikotnika, odstranitev ni dovoljena
            if (0 <= round(u, 10) <= 1 and 0 <= round(v, 10) <= 1) and round(u + v, 9) <= 1:
                return False

    return True