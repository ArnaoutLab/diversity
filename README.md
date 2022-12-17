# diversity: similarity- and frequency-sensitive diversity

![Tests](https://github.com/Elliot-D-Hill/diversity/actions/workflows/tests.yml/badge.svg)
[![Python 3.9](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-390/)

- [diversity: similarity- and frequency-sensitive diversity](#diversity-similarity--and-frequency-sensitive-diversity)
- [About](#about)
- [Usage and Examples](#usage-and-examples)
  - [Frequency-sensitive metacommunity from a dataframe or array](#frequency-sensitive-metacommunity-from-a-dataframe-or-array)
  - [Similarity-sensitive metacommunity from a dataframe or array](#similarity-sensitive-metacommunity-from-a-dataframe-or-array)
  - [Similarity-sensitive metacommunity from a file](#similarity-sensitive-metacommunity-from-a-file)
  - [Similarity-sensitive metacommunity from a function](#similarity-sensitive-metacommunity-from-a-function)
  - [Diversity measures](#diversity-measures)
  - [Command line interface](#command-line-interface)
- [Background](#background)
  - [Diversity indices](#diversity-indices)
  - [Partitioned diversity](#partitioned-diversity)
  - [Frequency-sensitive diversity](#frequency-sensitive-diversity)
  - [Similarity-sensitive diversity](#similarity-sensitive-diversity)
  - [One package to calculate them all](#one-package-to-calculate-them-all)
  - [Alternatives](#alternatives)


# About

The `diversity` package calculates similarity-sensitive and frequency-sensitive diversity measures.

**Supported diversity measures**: alpha, beta, rho, gamma, normalized alpha, normalized beta, and normalized rho.

**Supported diversity indices**: species richness, Shannon index, Simpson index, Berger-Parker index, and anything in between (Hill numbers).

For a rigorous mathematical treatment of the diversity measures and indices `diversity` can calculate see [Reeve et al., 2014](https://arxiv.org/abs/1404.6520). A brief informal discussion can be found in the [background](#background) section.

<!-- # Installation

`diversity` requires python version 3.10 or higher.

```bash
pip install diversity
``` -->


# Usage and Examples

The examples here use aggregated data from the [Palmer penguins dataset](https://github.com/allisonhorst/palmerpenguins).

```python
from diversity.metacommunity import Metacommunity
import pandas as pd
import numpy as np
```

The first step to calculating diversity is to create a `Metacommunity` object. We can create either a frequency-sensitive or similarity-sensitive `Metacommunity` object.

## Frequency-sensitive metacommunity from a dataframe or array

To create a frequency-sensitive metacommunity, we need a subcommunity-by-species counts table, where rows are unique species, columns are subcommunities, and the elements are species counts.

```python
counts = pd.DataFrame(
    {
      "Biscoe":    [44, 0, 120], 
      "Dream":     [55, 68, 0], 
      "Torgersen": [47, 0, 0]
    },
    index=["Adelie", "Chinstrap", "Gentoo"],
)
```

|           | Biscoe | Dream | Torgersen |
| :-------- | -----: | ----: | --------: |
| Adelie    |     44 |    55 |        47 |
| Chinstrap |      0 |    68 |         0 |
| Gentoo    |    120 |     0 |         0 |

Note: we include an index with the species names here for illustration, but in general, an index is not required for the counts (or similarity matrix).

Next we create a `Metacommunity` object from the counts table.

```python
metacommunity = Metacommunity(counts)
```

## Similarity-sensitive metacommunity from a dataframe or array

For similarity-sensitive diversity, we must also supply a similarity matrix to `Metacommunity` in addition to the counts table. The columns and rows of the similarity matrix must be in the same order as the rows of the counts table.

```python
similarity_matrix = pd.DataFrame(
    {
        "Adelie":    [1.000000, 0.347385, 0.222998],
        "Chinstrap": [0.347385, 1.000000, 0.258256],
        "Gentoo":    [0.222998, 0.258256, 1.000000],
    },
    index=["Adelie", "Chinstrap", "Gentoo"],
)
```

|           |   Adelie | Chinstrap |   Gentoo |
| :-------- | -------: | --------: | -------: |
| Adelie    |        1 |  0.347385 | 0.222998 |
| Chinstrap | 0.347385 |         1 | 0.258256 |
| Gentoo    | 0.222998 |  0.258256 |        1 |


```python
metacommunity = Metacommunity(counts, similarity=similarity_matrix)
```

## Similarity-sensitive metacommunity from a file

For medium sized datasets, the similarity matrix may not fit in RAM. To avoid loading the entire matrix into RAM, you can pass a filepath to the `similarity` argument to read a file from a hard disk drive.

```python
metacommunity = Metacommunity(counts, similarity='similarity_matrix.csv', chunk_size=100)
```

## Similarity-sensitive metacommunity from a function
For large datasets, the similarity matrix may not fit on the disk, in which case it can be constructed and processed in chunks by passing a similarity function to `similarity` and an array of features to `X`. Each row of `X` represents the features of a single species.
```python
X = np.array([[1, 2], [3, 5], [7, 11]])

def similarity_function(species_i, species_j):
  return 1 / (1 + np.norm(species_i, species_j))

metacommunity = Metacommunity(counts, similarity=similarity_function, X=X, chunk_size=100)
```

## Diversity measures
 
Once a `Metacommunty` object has been initialized, we can calculate all diversity measures for each subcommunity for a given viewpoint. Here we calculate similarity sensitive-species richness (viewpoint = 0).


```python
metacommunity.subcommunities_to_dataframe(viewpoint=0)
```

|      | community | viewpoint | alpha |  rho | beta | gamma | normalized_alpha | normalized_rho | normalized_beta |
| ---: | :-------- | --------: | ----: | ---: | ---: | ----: | ---------------: | -------------: | --------------: |
|    0 | Biscoe    |         0 |  3.15 | 1.70 | 0.59 |  1.89 |             1.55 |           0.84 |            1.20 |
|    1 | Dream     |         0 |  4.02 | 2.07 | 0.48 |  1.99 |             1.48 |           0.76 |            1.31 |
|    2 | Torgersen |         0 |  7.11 | 4.18 | 0.24 |  1.70 |             1.00 |           0.59 |            1.70 |


We can also compute all metacommunity measures at once. 
```python
metacommunity.subcommunities_to_dataframe(viewpoint=0)
```

|      | community     | viewpoint | alpha |  rho | beta | gamma | normalized_alpha | normalized_rho | normalized_beta |
| ---: | :------------ | --------: | ----: | ---: | ---: | ----: | ---------------: | -------------: | --------------: |
|    0 | metacommunity |         0 |  4.03 | 2.19 | 0.50 |  1.90 |             1.45 |           0.77 |            1.31 |


Individual diversity measures for subcommunities can also be calculated, like so:

```python
metacommunity.subcommunity_diversity(viewpoint=2, measure='alpha')

array([2.93063044, 4.00900135, 7.10638298])
```

and for the metacommunity:

```python
metacommunity.metacommunity_diversity(viewpoint=2, measure='beta')

0.48236433045721444
```


## Command line interface

`diversity` can also be used from the command-line as a module (via `python -m`).

The example below uses the `penguin_counts.csv` and `penguin_similarity_matrix.csv` files, which can be downloaded from this [gist](https://gist.github.com/Elliot-D-Hill/f6e9db0aebe561a363e8758c72a0acfc).

```bash
python -m diversity -i penguin_counts.csv -s penguin_similarity_matrix.csv -v 0 1 inf
```

This command produces subcommunity diversity measures for the viewpoint
parameter values 0, 1, and infinity:

|      | community     | viewpoint |  alpha |    rho |   beta |  gamma | normalized_alpha | normalized_rho | normalized_beta |
| ---: | :------------ | --------: | -----: | -----: | -----: | -----: | ---------------: | -------------: | --------------: |
|    0 | Biscoe        |         0 |  3.149 | 1.7035 |  0.587 | 1.8929 |           1.5462 |         0.8365 |          1.1955 |
|    1 | Dream         |         0 | 4.0194 | 2.0671 | 0.4838 | 1.9939 |           1.4802 |         0.7612 |          1.3137 |
|    2 | Torgersen     |         0 | 7.1064 | 4.1783 | 0.2393 | 1.7008 |                1 |          0.588 |          1.7008 |
|    3 | metacommunity |         0 | 4.0264 | 2.1856 | 0.5001 | 1.9031 |           1.4451 |         0.7738 |          1.3101 |
|    4 | Biscoe        |         1 | 3.0279 | 1.6028 | 0.6239 | 1.8891 |           1.4867 |          0.787 |          1.2707 |
|    5 | Dream         |         1 | 4.0142 | 2.0314 | 0.4923 | 1.9761 |           1.4783 |         0.7481 |          1.3367 |
|    6 | Torgersen     |         1 | 7.1064 | 4.1783 | 0.2393 | 1.7008 |                1 |          0.588 |          1.7008 |
|    7 | metacommunity |         1 | 3.7877 | 2.0014 | 0.4997 | 1.8925 |           1.4031 |         0.7414 |          1.3488 |
|    8 | Biscoe        |       inf |  2.573 | 1.3105 | 0.7631 | 1.7008 |           1.2634 |         0.6435 |           1.554 |
|    9 | Dream         |       inf | 3.8344 | 1.7187 | 0.5818 | 1.7008 |           1.4121 |         0.6329 |          1.5799 |
|   10 | Torgersen     |       inf | 7.1064 | 4.1783 | 0.2393 | 1.7008 |                1 |          0.588 |          1.7008 |
|   11 | metacommunity |       inf |  2.573 | 1.3105 | 0.2393 | 1.7008 |                1 |          0.588 |           1.554 |

Notes:
- The input filepath (`-i`) and the similarity matrix filepath (`-s`)
  Can be URLs to data files hosted on the web
- You can use .csv or .tsv for input files, but output is tab-delimited
- Output can be piped (piping will not include log statements in the output)
- Viewpoint parameter values of 100 or larger are treated like infinity
- For further options execute `python -m diversity -h`


# Background

## Diversity indices

A ***community*** is a collection of elements called ***individuals***, each of which is assigned a label called its ***species***, where multiple individuals may have the same species. A ***diversity index*** is a statistic associated with a community, which describes how much the species of its individuals vary. For example, a community of many individuals of the same species has a very low diversity whereas a community with multiple species and the same amount of individuals per species has a high diversity.


## Partitioned diversity

Some diversity indices compare the diversities of subsets of a community with respect to the overall community. The subsets are called ***subcommunities***, while the overall community is called a ***metacommunity***. For example, two communities with the same frequency distribution across two disjoint sets of the same number of unique species each comprise half of the combined metacommunity diversity. Examples of interpretations of some of these diversity indices include: the "representativeness" of a subcommunity of its metacommunity, and the "contribution" of a subcommunity to its metacommunity's diversity. 


## Frequency-sensitive diversity

[In 1973, Hill introduced a framework](https://doi.org/10.2307/1934352) which unifies commonly used diversity indices into a single parameterized family of diversity measures. The so-called ***viewpoint parameter*** can be thought of as the sensitivity to rare species. At one end of the spectrum, when the viewpoint parameter is set to 0, species frequency is ignored entirely, and only the number of distinct species matters, while at the other end of the spectrum, when the viewpoint parameter is set to ∞, only the highest frequency species in a community is considered by the corresponding diversity measure. Common diversity measures such as ***species richness***, ***Shannon entropy***, the ***Gini-Simpson index***, and the ***Berger-Parker index*** have simple and natural relationships with Hill's indices at different values for the viewpoint parameter (0, 1, 2, ∞, respectively).


## Similarity-sensitive diversity

In addition to being sensitive to frequency, it often makes sense to account for similarity in a diversity measure. For example, a community of 2 different types of rodents, may be considered less diverse than a community where one of the rodent species was replaced by the same number of individuals of a bird species. [Reeve et al.](https://arxiv.org/abs/1404.6520) and [Leinster and Cobbold](https://doi.org/10.1890/10-2402.1) present a general mathematically rigorous way of incorporating similarity measures into Hill's framework. The result is a family of similarity-sensitive diversity indices parameterized by the same viewpoint parameter as well as the similarity function used for the species in the meta- or subcommunities of interest. These similarity-sensitive diversity measures account for both the pairwise similarity between all species and their frequencies.


## One package to calculate them all

The `diversity` package is able to calculate all of the similarity- and frequency-sensitive subcommunity and metacommunity diversity measures described in [Reeve et al.](https://arxiv.org/abs/1404.6520). See the paper for more in-depth information on their derivation and interpretation.

## Alternatives

Diversity can be defined in various ways and software calculating the various diversity measures exists. To date, we know of no other python package that implements the similarity-sensitive partitioned diversity measures defined by [Reeve at al.](https://arxiv.org/abs/1404.6520). An [R package](https://github.com/boydorr/rdiversity) and a [julia package](https://github.com/EcoJulia/Diversity.jl) exist. However, both packages require the species similarities to be stored in the form of a matrix in memory. That approach does not scale to the amount of species in some applications, such as immune repertoires. `diversity` allows the user to store the similarity matrix in a file or construct it on the fly, allowing for larger datasets to be analyzed.

