# Port of iTOS to Python
This is a port of
[iTOS](https://cran.r-project.org/web/packages/iTOS/index.html) to
Python, from R. The R package was created by Paul Rosenbaum in support
of his book, "[An Introduction to the Theory of Observational
Studies](http://www-stat.wharton.upenn.edu/~rosenbap/iTOS.html)".

I'm more comfortable with python than R, so I had ChatGPT port the R
package to python. No warrant is made for correctness. Any issues with
this python package are my fault, certainly not Professor Rosenbaum's.

Here is the description from the iTOS R package:

Supplements for a book, "iTOS" = "Introduction to the Theory of
Observational Studies." Data sets are 'aHDL' from Rosenbaum (2023a)
<doi:10.1111/biom.13558> and 'bingeM' from Rosenbaum (2023b)
<doi:10.1111/biom.13921>. The function makematch() uses two-criteria
matching from Zhang et al. (2023) <doi:10.1080/01621459.2021.1981337>
to create the matched data 'bingeM' from 'binge'. The makematch()
function also implements optimal matching (Rosenbaum (1989)
<doi:10.2307/2290079>) and matching with fine or near-fine balance
(Rosenbaum et al. (2007) <doi:10.1198/016214506000001059> and Yang et
al (2012) <doi:10.1111/j.1541-0420.2011.01691.x>). The book makes use
of two other R packages, 'weightedRank' and 'tightenBlock'.
