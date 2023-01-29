# diversity: partitioned frequency- and similarity-sensitive diversity in Python

![Tests](https://github.com/Elliot-D-Hill/diversity/actions/workflows/tests.yml/badge.svg)
[![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/release/python-380/)

- [diversity: partitioned frequency- and similarity-sensitive diversity in Python](#diversity-partitioned-frequency--and-similarity-sensitive-diversity-in-python)
- [About](#about)
- [Usage and Examples](#usage-and-examples)
  - [Frequency-sensitive metacommunity from a dataframe or array](#frequency-sensitive-metacommunity-from-a-dataframe-or-array)
  - [Frequency- and similarity-sensitive metacommunity from a dataframe or array](#frequency--and-similarity-sensitive-metacommunity-from-a-dataframe-or-array)
  - [Diversity measures](#diversity-measures)
  - [Frequency- and similarity-sensitive metacommunity from a file](#frequency--and-similarity-sensitive-metacommunity-from-a-file)
  - [Frequency- and similarity-sensitive metacommunity from a function](#frequency--and-similarity-sensitive-metacommunity-from-a-function)
  - [Command line interface](#command-line-interface)
- [Background](#background)
  - [Diversity indices](#diversity-indices)
  - [Partitioned diversity](#partitioned-diversity)
  - [Frequency-sensitive diversity](#frequency-sensitive-diversity)
  - [Similarity-sensitive diversity](#similarity-sensitive-diversity)
  - [One package to calculate them all](#one-package-to-calculate-them-all)
- [Alternatives](#alternatives)

# About

The `diversity` package calculates partitioned frequency- and similarity-sensitive diversity measures for a given metacommunity and its subcommunities.

**Supported subcommunity diversity measures**:

  - $\alpha$ - estimate of naive-community metacommunity diversity
  - $\bar{\alpha}$ - diversity of subcommunity j in isolation
  - $\rho$ - redundancy of subcommunity j
  - $\bar{\rho}$ - representativeness of subcommunity j
  - $\beta$ - distinctiveness of subcommunity j
  - $\bar{\beta}$ - estimate of the effective number of distinct subcommunities
  - $\gamma$ - contribution per individual toward metacommunity diversity


**Supported metacommunity diversity measures**:
  - $A$ - naive-community metacommunity diversity
  - $\bar{A}$ - average diversity of subcommunities
  - $R$ - average redundancy of subcommunities
  - $\bar{R}$ - average representativeness of subcommunities
  - $B$ - average distinctiveness of subcommunities
  - $\bar{B}$ - effective number of distinct subcommunities
  - $G$ - metacommunity diversity

For a more rigorous description of the diversity measures `diversity` can calculate see [Reeve et al., 2014](https://arxiv.org/abs/1404.6520). A brief informal discussion can be found in the [background](#background) section.

<!-- # Installation

`diversity` requires python version 3.8 or higher.

```bash
pip install diversity
``` -->

# Usage and Examples

The examples here use aggregated data from the [Palmer penguins dataset](https://github.com/allisonhorst/palmerpenguins).

```python
from diversity import Metacommunity
import pandas as pd
import numpy as np
```

To calculate diversity, the first step is to create a `Metacommunity` object.

## Frequency-sensitive metacommunity from a dataframe or array

To create a frequency-sensitive metacommunity, we need a subcommunity-by-species counts table, where rows are unique species and columns are subcommunities, and the elements are species counts. In this example, the rows are penguin species and the columns are islands where the penguins were observed.

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

Next we create a `Metacommunity` object by passing it the counts table.

```python
metacommunity = Metacommunity(counts)
```

## Frequency- and similarity-sensitive metacommunity from a dataframe or array

For frequency- and similarity-sensitive diversity, we must also supply a species similarity matrix to `Metacommunity` in addition to the counts table. The rows and columns of the similarity matrix must be in the same order as the rows of the counts table for valid results.

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

## Diversity measures
 
Once a `Metacommunty` object has been initialized, we can calculate all diversity measures for the metacommunity and each subcommunity for a set of viewpoints.

```python
metacommunity.to_dataframe(viewpoint=[0, 1, np.inf])
```

|      | community     | viewpoint | alpha |  rho | beta | gamma | normalized_alpha | normalized_rho | normalized_beta |
| ---: | :------------ | --------: | ----: | ---: | ---: | ----: | ---------------: | -------------: | --------------: |
|    0 | metacommunity |      0.00 |  4.03 | 2.19 | 0.50 |  1.90 |             1.45 |           0.77 |            1.31 |
|    1 | Biscoe        |      0.00 |  3.15 | 1.70 | 0.59 |  1.89 |             1.55 |           0.84 |            1.20 |
|    2 | Dream         |      0.00 |  4.02 | 2.07 | 0.48 |  1.99 |             1.48 |           0.76 |            1.31 |
|    3 | Torgersen     |      0.00 |  7.11 | 4.18 | 0.24 |  1.70 |             1.00 |           0.59 |            1.70 |
|    4 | metacommunity |      1.00 |  3.79 | 2.00 | 0.50 |  1.89 |             1.40 |           0.74 |            1.35 |
|    5 | Biscoe        |      1.00 |  3.03 | 1.60 | 0.62 |  1.89 |             1.49 |           0.79 |            1.27 |
|    6 | Dream         |      1.00 |  4.01 | 2.03 | 0.49 |  1.98 |             1.48 |           0.75 |            1.34 |
|    7 | Torgersen     |      1.00 |  7.11 | 4.18 | 0.24 |  1.70 |             1.00 |           0.59 |            1.70 |
|   12 | metacommunity |       inf |  2.57 | 1.31 | 0.24 |  1.70 |             1.00 |           0.59 |            1.55 |
|   13 | Biscoe        |       inf |  2.57 | 1.31 | 0.76 |  1.70 |             1.26 |           0.64 |            1.55 |
|   14 | Dream         |       inf |  3.83 | 1.72 | 0.58 |  1.70 |             1.41 |           0.63 |            1.58 |
|   15 | Torgersen     |       inf |  7.11 | 4.18 | 0.24 |  1.70 |             1.00 |           0.59 |            1.70 |

Individual diversity measures for subcommunities can also be calculated, like so:

```python
metacommunity.subcommunity_diversity(viewpoint=2, measure='alpha')

array([2.93063044, 4.00900135, 7.10638298])
```

and likewise for the metacommunity:

```python
metacommunity.metacommunity_diversity(viewpoint=2, measure='beta')

0.48236433045721444
```

## Frequency- and similarity-sensitive metacommunity from a file

For medium sized datasets, the similarity matrix may not fit in RAM. To avoid loading the entire matrix into RAM, you can pass a filepath to the `similarity` argument to read a file from a hard disk drive.

```python
metacommunity = Metacommunity(counts, similarity='similarity_matrix.csv', chunk_size=100)
```

## Frequency- and similarity-sensitive metacommunity from a function
For large datasets, the similarity matrix may not fit on the disk, in which case it can be constructed and processed in chunks by passing a similarity function to `similarity` and an array of features to `X`. Each row of `X` represents the feature values of a species.
```python
X = np.array([
  [1, 2], 
  [3, 4], 
  [5, 6]
])

def similarity_function(species_i, species_j):
  return 1 / (1 + np.norm(species_i, species_j))

metacommunity = Metacommunity(counts, similarity=similarity_function, X=X, chunk_size=100)
```

## Command line interface

`diversity` can also be used from the command-line as a module (via `python -m`).

The example below uses the `penguin_counts.csv` and `penguin_similarity_matrix.csv` files, which can be downloaded from this [gist](https://gist.github.com/Elliot-D-Hill/f6e9db0aebe561a363e8758c72a0acfc).

```bash
python -m diversity -i penguin_counts.csv -s penguin_similarity_matrix.csv -v 0 1 inf
```

This command produces metacommunity and subcommunity diversity measures for the viewpoint
parameter values 0, 1, and infinity.

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

Some diversity indices compare the diversities of subsets of a community with respect to the overall community. The subsets are called ***subcommunities***, while the overall community is called a ***metacommunity***. For example, two subcommunities with the same frequency distribution but no shared species each comprise half of the combined metacommunity diversity.

## Frequency-sensitive diversity

[In 1973, Hill introduced a framework](https://doi.org/10.2307/1934352) which unifies commonly used diversity indices into a single parameterized family of diversity measures. The so-called ***viewpoint parameter*** can be thought of as the sensitivity to rare species. At one end of the spectrum, when the viewpoint parameter is set to 0, species frequency is ignored entirely, and only the number of distinct species matters, while at the other end of the spectrum, when the viewpoint parameter is set to $\infty$, only the highest frequency species in a community is considered by the corresponding diversity measure. Common diversity measures such as ***species richness***, ***Shannon entropy***, the ***Gini-Simpson index***, and the ***Berger-Parker index*** have simple and natural relationships with Hill's indices at different values for the viewpoint parameter (0, 1, 2, $\infty$, respectively).

## Similarity-sensitive diversity

In addition to being sensitive to frequency, it often makes sense to account for similarity in a diversity measure. For example, a community of two different types of rodents, may be considered less diverse than a community where one of the rodent species was replaced by the same number of individuals of a bird species. [Reeve et al.](https://arxiv.org/abs/1404.6520) and [Leinster and Cobbold](https://doi.org/10.1890/10-2402.1) present a general mathematically rigorous way of incorporating similarity measures into Hill's framework. The result is a family of similarity-sensitive diversity indices parameterized by the same viewpoint parameter as well as the similarity function used for the species in the meta- or subcommunities of interest. These similarity-sensitive diversity measures account for both the pairwise similarity between all species and their frequencies.

## One package to calculate them all

The `diversity` package is able to calculate all of the similarity- and frequency-sensitive subcommunity and metacommunity diversity measures described in [Reeve et al.](https://arxiv.org/abs/1404.6520). See the paper for more in-depth information on their derivation and interpretation.

# Alternatives

To date, we know of no other python package that implements the partitioned frequency- and similarity-sensitive diversity measures defined by [Reeve at al.](https://arxiv.org/abs/1404.6520). However, there is a [R package](https://github.com/boydorr/rdiversity) and a [Julia package](https://github.com/EcoJulia/Diversity.jl).

