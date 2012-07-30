gdt
===

Generalized Distance Transform

**C_gdt.c** implements a Squared Euclidean Distance Transform.
Ie. `v(p) = min_q (u(q) + |p-q|^2)`, where `|.|^2` is the squared euclidean
distance.

On top of that the python function **prob_dt** implements what I call a
Probabilistic Distance Transform. The input is a map of probabilities, and
you want to smear the probabilities around. More specifically you want them
to decay according to a Guassian distribution with standard deviation sigma.
