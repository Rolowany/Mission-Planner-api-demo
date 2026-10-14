# Prosty autorski planner misji

Idee tego planera wymyśliłem samemu spacerując po parku. Interesuje się optymalizacją, więc chciałem jakoś wykorzystać posiadaną w tym zakresie wiedzę. Na wejściu dostajemy jakiś zestaw punktów, które musimy odwiedzić. Oczywiście, im krótsza ścieżka tym lepiej (mniej paliwa, krótszy czas). Na starcie rozwiązujemy, więc problem komiwojażera. Jednakże by nie było tak nudno, część punktów (oczwyiście nie wszystkie bo problem stałby się trywialny) musi być odwiedzona w określonej kolejności. Na początku, więc odpalamy algorytm [symulowanego wyżarzania] (https://en.wikipedia.org/wiki/Simulated_annealing). Jak już mamy trasę, dron rusza w drogę. Równolegle z innego pliku wczytujemy położenie przeszkód, które statek ma ominąć. Program symuluje zachowanie lidara, który wykrywa przeszkody, a statek omija je.

# Jak odpalić?

W tle należy odpalić Mission Planner. Pracuję na linuxie, więc używałem mono. Wybrać miejsce startowe z danego pliku (nie wiem jak to zrobić by się samo ustawiało z poziomy Pythona). Wybrać dron i czekać na połączenie. Następnie odpalamy skrypt. 

```uv pip install -r requirements.txt```

## To pokaże argumenty, którymi można się bawić

```python -m src.main -h```

## Sama optymalizacja trasy bez przeszkód

```python -m src.main missions/blonia.json --scenario scenarios/no_obstacles.json```

## Proste wyminięcie drzewa

```python -m src.main missions/avoidance_test.json --scenario scenarios/single_tree.json```
