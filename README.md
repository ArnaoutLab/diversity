# diversity

Calculates similarity-sensitive diversity indices.

* [About](#about)
* [Installation](#installation)
* [Usage and Examples](#usage-and-examples)
* [Background](#background)
    + [Diversity indices](#diversity-indices)
    + [Partitioned diversity](#partitioned-diversity)
    + [Frequency-sensitive diversity](#frequency-sensitive-diversity)
    + [Similarity-sensitive diversity](#similarity-sensitive-diversity)
    + [One package to calculate them all](#one-package-to-calculate-them-all)
* [Alternatives](#alternatives)

## About

For a rigorous mathematical treatment of the diversity indices
calculated by `diversity`, see
[Reeve et al., 2014](https://arxiv.org/abs/1404.6520). A brief informal
discussion can be found in the [background section](#background).

## Installation

To install, execute:
```bash
pip install diversity
```

**Dependencies:**
- pytest >= 6.2.5
- numpy >= 1.22.0
- pandas >= 1.3.5

To run tests, execute:
```bash
pytest --pyargs metacommunity
```

## Usage and Examples

## Background

### Diversity indices

A ***community*** is a collection of elements called ***individuals***,
each of which is assigned a label called its ***species***, where
multiple individuals may have the same species. A ***diversity index***
is a statistic associated with a community, which describes how much the
species of its individuals vary. For example, a community of many
individuals of the same species has a very low diversity whereas a
community with multiple species and the same amount of individuals per
species has a high diversity.

### Partitioned diversity

Some diversity indices compare the diversities of subsets of a community
with respect to the overall community. The subsets are called
***subcommunities***, while the overall community is called a
***metacommunity***. For example, two communities with the same
frequency distribution across two disjoint sets of the same number of
unique species each comprise half of the combined metacommunity
diversity. Examples of interpretations of some of these diversity
indices include: the "representativeness" of a subcommunity of its
metacommunity, and the "contribution" of a subcommunity to its
metacommunity's diversity. 

### Frequency-sensitive diversity

[In 1973, Hill introduced a framework](https://doi.org/10.2307/1934352)
which unifies commonly used diversity indices into a single
parameterized family of diversity measures. The so-called ***viewpoint
parameter*** can be thought of as the sensitivity to rare species. At
one end of the spectrum, when the viewpoint parameter is set to 0,
species frequency is ignored entirely, and only the number of distinct
species matters, while at the other end of the spectrum, when the
viewpoint parameter is set to ∞, only the highest frequency species in a
community is considered by the corresponding diversity measure. Common
diversity measures such as ***species richness***, ***Shannon
entropy***, the ***Gini-Simpson index***, and the ***Berger-Parker
index*** have simple and natural relationships with Hill's indices at
different values for the viewpoint parameter (0, 1, 2, ∞, respectively).

### Similarity-sensitive diversity

In addition to being sensitive to frequency, it often makes sense to
account for similarity in a diversity measure. For example, a community
of 2 different types of rodents, may be considered less diverse as the
same community where one of the rodent types was replaced by the same
number of individuals of a bird species.
[Reeve et al.](https://arxiv.org/abs/1404.6520)
[Leinster and Cobbold](https://doi.org/10.1890/10-2402.1) present a
general mathematically rigorous way of incorporating similarity measures
into Hill's framework. The result is a family of similarity-sensitive
diversity indices parameterized by the same viewpoint parameter as well
as the similarity function used for the species in the meta- or
subcommunities of interest. As opposed to accounting for distinct
species and their frequency, these similarity-sensitive diversity
measures can be interpreted as accounting for different possibly
overlapping clusters of mutually similar species and their combined
frequencies.

### One package to calculate them all

The `diversity` package is able to calculate all of the similarity- and
frequency-sensitive subcommunity and metacommunity diversity measures
described in [Reeve et al.](https://arxiv.org/abs/1404.6520). See the
paper for more in-depth information on their derivation and
interpretation.

## Alternatives

To date, we know of no other python package that implements the
similarity-sensitive diversity measures calculated by `diversity`. An
[R package](https://github.com/boydorr/rdiversity) and a
[julia package](https://github.com/EcoJulia/Diversity.jl) exist.
However, both packages require the species similarities to be stored in
the form of a matrix in memory. That approach does not scale to the
amount of species in some applications, such as immune repertoires.
`diversity` allows the user to store the similarity matrix in a file, or
simply provide a python function that computes the similarities on the
fly.
