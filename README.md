gdt
===

I needed a Generalized Distance Transform in my python program and didn't find
any open source implementation that did what I needed. So in case anybody needs
an Squared Euclidean Distance Transform implementationt that
* can deal with `inf`
* has runtime _O(n)_
here you go.

Pull requests are welcome, of course.

Generalized Distance Transform
------------------------------

**C_gdt.c** implements a Squared Euclidean Distance Transform.  Ie. `v(p) =
min_q (u(q) + |p-q|^2)`, where `p` and `q` are 2D coordinates in the image,
`|.|^2` is the squared euclidean distance, `u(.)` is the original image,
and `v(.)` is the output image.


Probabilistic Distance transform
--------------------------------

On top of that the python function **prob_dt** implements what I call a
Probabilistic Distance Transform. The input is a map of probabilities, and
you want to smear the probabilities around. More specifically you want them
to decay according to a Guassian distribution with standard deviation sigma.

The function is `prob_dt(x) = max_y (p(y) * exp(-|x-y|^2 / (2 * sigma^2)))`,
which I solve in logspace: `log p(y) - |x-y|^2 / (2 * sigma^2)`. You can
already see how the gaussian turned into a squared euclidean distance. Now we
just have to isolate the distance term. The final result is `prob_dt(x) =
exp(1/(2*sigma^2) * gdt(log(p(x)) * 2 * sigma^2))`.
