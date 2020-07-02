from edu_tools import formatuj_wynik, depol


def test_formatuj_wynik():
    assert formatuj_wynik() == ''
    assert formatuj_wynik(pnts=np.nan) == 'brak'
    assert formatuj_wynik(pnts=8, max_pnts=10) == '8 / 10 (80%)'
    assert formatuj_wynik(perc=8) == '8%'
    assert formatuj_wynik(perc=10, max_pnts=10) == '1 / 10 (10%)'


def test_depol():
    txt1 = 'Wszelkie szyszki, które garną się do łyżki, jakiś jeż zje.'
    txt2 = 'Wszelkie szyszki, ktore garna sie do lyzki, jakis jez zje.'
    assert depol(txt1) == txt2

    # this will fail
    assert depol('Świt') == 'Swit'
