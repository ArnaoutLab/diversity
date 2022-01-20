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

For a rigorous mathematical treatment of the diversity indices calculated by `diversity`, see [Reeve et al., 2014](https://arxiv.org/abs/1404.6520). A brief informal discussion can be found in the [background section](#background).

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

For calculating diversities of [the microbiomes from 4 cyclists]() ([FIXME](FIXME)) from the command line you need the following files:

`metagenome_counts.csv`
```
subcommunity,species,counts
subcommunity_1,Akkermansia muciniphila,4394
subcommunity_1,Faecalibacterium prausnitzii,1807
...
subcommunity_4,Blautia coccoides,1
```

and

`metagenome_similarity_matrix.csv`
```
Akkermansia muciniphila,Faecalibacterium prausnitzii,...,Shigella dysenteriae
1.0,0.6671415709437735,...,0.6093549494839958
...
0.6093549494839958,0.6185898023961985,...,1.0
```

and execute
```bash
python -m diversity -i metagenome_counts.csv -s metagenome_similarity_matrix.csv
```

This is what is printed to the terminal window:
```
2022-01-19T14:02:44-0500 (MainProcess, MainThread) INFO python3.9.4 /Users/jasper/diversity/src/diversity/__main__.py -i metagenome_counts.csv -s metagenome_similarity_matrix.csv -v 0 1 2 inf
community viewpoint alpha rho beta gamma normalized_alpha normalized_rho normalized_beta
subcommunity_1 0.00 5.6056 3.8792 0.2578 1.4479 1.4008 0.9694 1.0316
subcommunity_2 0.00 5.7490 3.9715 0.2518 1.4480 1.4375 0.9930 1.0070
subcommunity_3 0.00 5.5985 3.9352 0.2541 1.4222 1.3995 0.9837 1.0165
subcommunity_4 0.00 5.3856 3.8230 0.2616 1.4068 1.3469 0.9561 1.0459
metacommunity 0.00 5.5847 3.9022 0.2563 1.4312 1.3962 0.9756 1.0253
subcommunity_1 1.00 5.6017 3.8719 0.2583 1.4468 1.3998 0.9675 1.0335
subcommunity_2 1.00 5.7449 3.9703 0.2519 1.4470 1.4365 0.9927 1.0073
subcommunity_3 1.00 5.5895 3.9347 0.2541 1.4206 1.3973 0.9836 1.0167
subcommunity_4 1.00 5.3633 3.8167 0.2620 1.4052 1.3413 0.9545 1.0477
metacommunity 1.00 5.5732 3.8980 0.2565 1.4298 1.3933 0.9745 1.0262
subcommunity_1 2.00 5.5979 3.8646 0.2588 1.4456 1.3989 0.9657 1.0355
subcommunity_2 2.00 5.7410 3.9692 0.2519 1.4459 1.4355 0.9925 1.0076
subcommunity_3 2.00 5.5808 3.9342 0.2542 1.4189 1.3951 0.9835 1.0168
subcommunity_4 2.00 5.3421 3.8106 0.2624 1.4037 1.3360 0.9530 1.0493
metacommunity 2.00 5.5617 3.8937 0.2568 1.4283 1.3904 0.9734 1.0271
subcommunity_1 inf 5.2934 3.5613 0.2808 1.3452 1.3228 0.8899 1.1237
subcommunity_2 inf 5.4517 3.8544 0.2594 1.3452 1.3632 0.9638 1.0376
subcommunity_3 inf 5.2445 3.8535 0.2595 1.3452 1.3110 0.9633 1.0381
subcommunity_4 inf 4.9728 3.6221 0.2761 1.3452 1.2436 0.9058 1.1040
metacommunity inf 4.9728 3.5613 0.2594 1.3452 1.2436 0.8899 1.0376
2022-01-19T14:02:44-0500 (MainProcess, MainThread) INFO Done!
2022-01-19T14:02:44-0500 (MainProcess, MainThread) INFO process shutting down
```

Notes:
- you can use .csv or .tsv for input files, but output is tab-delimited
- piping output will not include log statements
- for further options execute `python -m diversity -h`

## Background

### Diversity indices

A ***community*** is a collection of elements called ***individuals***, each of which is assigned a label called its ***species***, where multiple individuals may have the same species. A ***diversity index*** is a statistic associated with a community, which describes how much the species of its individuals vary. For example, a community of many individuals of the same species has a very low diversity whereas a community with multiple species and the same amount of individuals per species has a high diversity.

### Partitioned diversity

Some diversity indices compare the diversities of subsets of a community with respect to the overall community. The subsets are called ***subcommunities***, while the overall community is called a ***metacommunity***. For example, two communities with the same frequency distribution across two disjoint sets of the same number of unique species each comprise half of the combined metacommunity diversity. Examples of interpretations of some of these diversity indices include: the "representativeness" of a subcommunity of its metacommunity, and the "contribution" of a subcommunity to its metacommunity's diversity. 

### Frequency-sensitive diversity

[In 1973, Hill introduced a framework](https://doi.org/10.2307/1934352) which unifies commonly used diversity indices into a single parameterized family of diversity measures. The so-called ***viewpoint parameter*** can be thought of as the sensitivity to rare species. At one end of the spectrum, when the viewpoint parameter is set to 0, species frequency is ignored entirely, and only the number of distinct species matters, while at the other end of the spectrum, when the viewpoint parameter is set to ∞, only the highest frequency species in a community is considered by the corresponding diversity measure. Common diversity measures such as ***species richness***, ***Shannon entropy***, the ***Gini-Simpson index***, and the ***Berger-Parker index*** have simple and natural relationships with Hill's indices at different values for the viewpoint parameter (0, 1, 2, ∞, respectively).

### Similarity-sensitive diversity

In addition to being sensitive to frequency, it often makes sense to account for similarity in a diversity measure. For example, a community of 2 different types of rodents, may be considered less diverse as the same community where one of the rodent types was replaced by the same number of individuals of a bird species. [Reeve et al.](https://arxiv.org/abs/1404.6520) [Leinster and Cobbold](https://doi.org/10.1890/10-2402.1) present a general mathematically rigorous way of incorporating similarity measures into Hill's framework. The result is a family of similarity-sensitive diversity indices parameterized by the same viewpoint parameter as well as the similarity function used for the species in the meta- or subcommunities of interest. As opposed to accounting for distinct species and their frequency, these similarity-sensitive diversity measures can be interpreted as accounting for different possibly overlapping clusters of mutually similar species and their combined frequencies.

### One package to calculate them all

The `diversity` package is able to calculate all of the similarity- and frequency-sensitive subcommunity and metacommunity diversity measures described in [Reeve et al.](https://arxiv.org/abs/1404.6520). See the paper for more in-depth information on their derivation and interpretation.

## Alternatives

To date, we know of no other python package that implements the similarity-sensitive diversity measures calculated by `diversity`. An [R package](https://github.com/boydorr/rdiversity) and a [julia package](https://github.com/EcoJulia/Diversity.jl) exist. However, both packages require the species similarities to be stored in the form of a matrix in memory. That approach does not scale to the amount of species in some applications, such as immune repertoires. `diversity` allows the user to store the similarity matrix in a file, or simply provide a python function that computes the similarities on the fly.
