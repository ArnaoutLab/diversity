# diversity: similarity-sensitive diversity indices

![Tests](https://github.com/Elliot-D-Hill/diversity/actions/workflows/tests.yml/badge.svg)
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="138" height="20" role="img" aria-label="python: 3.8 | 3.9 | 3.10"><title>python: 3.8 | 3.9 | 3.10</title><linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="r"><rect width="138" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#r)"><rect width="49" height="20" fill="#555"/><rect x="49" width="89" height="20" fill="#007ec6"/><rect width="138" height="20" fill="url(#s)"/></g><g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110"><text aria-hidden="true" x="255" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="390">python</text><text x="255" y="140" transform="scale(.1)" fill="#fff" textLength="390">python</text><text aria-hidden="true" x="925" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="790">3.8 | 3.9 | 3.10</text><text x="925" y="140" transform="scale(.1)" fill="#fff" textLength="790">3.8 | 3.9 | 3.10</text></g></svg>
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

* [About](#about)
* [Installation](#installation)
* [Usage and Examples](#usage-and-examples)
    + [Python](#python)
        + [Similarity from array or dataframe](#similarity-from-array-or-dataframe)
        + [Similarity from file](#similarity-from-file)
        + [Similarity from function](#similarity-from-function)
    + [Command line interface](#command-line-interface)
        + [Metagenomics dataset](#metagenomics-dataset)
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

## Usage and Examples


### Python

```python
from diversity.metacommunity import make_metacommunity
from pandas import read_csv
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
```
The data for the following examples can be found at this [Gist](https://gist.github.com/Elliot-D-Hill/1200c74c83a4be1f63f9ef56b333bb99).

```python
gist_url = 'https://gist.github.com/Elliot-D-Hill/1200c74c83a4be1f63f9ef56b333bb99/raw/71e2c2f026b9a495023e6e71ee1dcc545cb53972/'
```


#### Similarity from array or dataframe

First we load the [Palmer penguins dataset](https://github.com/allisonhorst/palmerpenguins) (see [Gorman, Williams, and Fraser, 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0090081) for more details). To simplify our examples, we have preprocessed the dataset to include only the means of four features for the three penguin species (Adelie, Chinstrap, and Gentoo).


```python
penguin_features_filepath = gist_url + 'penguin_features.csv'
penguin_features = read_csv(penguin_features_filepath)
penguin_features.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>culmen_length_mm</th>
      <th>culmen_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>38.823973</td>
      <td>18.347260</td>
      <td>190.102740</td>
      <td>3706.164384</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chinstrap</td>
      <td>48.833824</td>
      <td>18.420588</td>
      <td>195.823529</td>
      <td>3733.088235</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gentoo</td>
      <td>47.542500</td>
      <td>15.002500</td>
      <td>217.233333</td>
      <td>5090.625000</td>
    </tr>
  </tbody>
</table>
</div>



Next, we load in the penguin subcommunity-species counts data.


```python
penguin_counts_filepath = gist_url + 'penguin_counts.csv'
penguin_counts = read_csv(penguin_counts_filepath)
penguin_counts.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subcommunity</th>
      <th>species</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Biscoe</td>
      <td>Gentoo</td>
      <td>120</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dream</td>
      <td>Chinstrap</td>
      <td>68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dream</td>
      <td>Adelie</td>
      <td>55</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Torgersen</td>
      <td>Adelie</td>
      <td>47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Biscoe</td>
      <td>Adelie</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>



First, we generate a similarity matrix from the penguin features by taking the euclidean distance between each row in the dataset. Then, we convert the distances to similarities by scaling between 0 and 1.


```python
species_order = penguin_features.pop('species')
distances = pdist(penguin_features)
penguin_distance_matrix = squareform(distances)
penguin_similarity_matrix = 1 / (1  + penguin_distance_matrix)
penguin_similarity_matrix
```

    array([[1.00000000e+00, 3.30156918e-02, 7.21626767e-04],
           [3.30156918e-02, 1.00000000e+00, 7.35991958e-04],
           [7.21626767e-04, 7.35991958e-04, 1.00000000e+00]])


Last, we create a metacommunity object from the counts, spcecies order, and similarity matrix.

```python
penguin_metacommunity = make_metacommunity(
    penguin_counts, 
    species_order=species_order, 
    similarity_matrix=penguin_similarity_matrix
)
```

We can calculate all diviersty measures at once for a given viewpoint.


```python
penguin_metacommunity.subcommunities_to_dataframe(viewpoint=0)
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>community</th>
      <th>viewpoint</th>
      <th>alpha</th>
      <th>rho</th>
      <th>beta</th>
      <th>gamma</th>
      <th>normalized_alpha</th>
      <th>normalized_rho</th>
      <th>normalized_beta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Biscoe</td>
      <td>0</td>
      <td>4.068632</td>
      <td>1.635146</td>
      <td>0.611566</td>
      <td>2.638069</td>
      <td>1.997771</td>
      <td>0.802886</td>
      <td>1.245507</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dream</td>
      <td>0</td>
      <td>5.253772</td>
      <td>1.735989</td>
      <td>0.576041</td>
      <td>3.539491</td>
      <td>1.934772</td>
      <td>0.639301</td>
      <td>1.564208</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Torgersen</td>
      <td>0</td>
      <td>7.106383</td>
      <td>3.155993</td>
      <td>0.316858</td>
      <td>2.251711</td>
      <td>1.000000</td>
      <td>0.444107</td>
      <td>2.251711</td>
    </tr>
  </tbody>
</table>
</div>



We can change the viewpoint parameter and recalculate diversity measures.


```python
penguin_metacommunity.subcommunities_to_dataframe(viewpoint=2)
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>community</th>
      <th>viewpoint</th>
      <th>alpha</th>
      <th>rho</th>
      <th>beta</th>
      <th>gamma</th>
      <th>normalized_alpha</th>
      <th>normalized_rho</th>
      <th>normalized_beta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Biscoe</td>
      <td>2</td>
      <td>3.351522</td>
      <td>1.233509</td>
      <td>0.810695</td>
      <td>2.615200</td>
      <td>1.645657</td>
      <td>0.605675</td>
      <td>1.651050</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dream</td>
      <td>2</td>
      <td>5.202916</td>
      <td>1.424577</td>
      <td>0.701963</td>
      <td>3.132199</td>
      <td>1.916044</td>
      <td>0.524620</td>
      <td>1.906143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Torgersen</td>
      <td>2</td>
      <td>7.106383</td>
      <td>3.155993</td>
      <td>0.316858</td>
      <td>2.251711</td>
      <td>1.000000</td>
      <td>0.444107</td>
      <td>2.251711</td>
    </tr>
  </tbody>
</table>
</div>



We can also compute all metacommunity diversity measures at once. 


```python
penguin_metacommunity.metacommunity_to_dataframe(viewpoint=0)
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>community</th>
      <th>viewpoint</th>
      <th>alpha</th>
      <th>rho</th>
      <th>beta</th>
      <th>gamma</th>
      <th>normalized_alpha</th>
      <th>normalized_rho</th>
      <th>normalized_beta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>metacommunity</td>
      <td>0</td>
      <td>4.932543</td>
      <td>1.886294</td>
      <td>0.557012</td>
      <td>2.915662</td>
      <td>1.834166</td>
      <td>0.692157</td>
      <td>1.504464</td>
    </tr>
  </tbody>
</table>
</div>



If we only want to compute a single measure, we call them individually.


```python
penguin_metacommunity.subcommunity_alpha(viewpoint=0)
```

    array([4.06863173, 5.25377155, 7.10638298])



#### Similarity from file

For large datasets, the similarity matrix may not fit in RAM. To avoid loading the entire matrix into RAM, use the similarity_matrix_filepath argument to read a file from a hard disk drive.

This example calculates diversity indices for the gut microbiomes of four cyclists. For our example, we use a subsample of this dataset. The full dataset can be found [here](https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP219106). See [Peterson, et al., 2017](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-017-0320-4) for more details.


```python
# read the files from gist
metagenome_similarity_filepath = gist_url + 'metagenome_similarity_matrix.csv'
metagenome_counts_filepath = gist_url + 'metagenome_counts.csv'
metagenome_similarity = read_csv(metagenome_similarity_filepath)
metagenome_counts = read_csv(metagenome_counts_filepath)

# save the similarity matrix file locally
similarity_filepath = 'metagenome_similarity_matrix.csv'
metagenome_similarity.to_csv(similarity_filepath, index=None)
```


```python
metagenome = make_metacommunity(
    metagenome_counts, 
    similarity_matrix_filepath=similarity_filepath
)
```


```python
metagenome.subcommunities_to_dataframe(viewpoint=0)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>community</th>
      <th>viewpoint</th>
      <th>alpha</th>
      <th>rho</th>
      <th>beta</th>
      <th>gamma</th>
      <th>normalized_alpha</th>
      <th>normalized_rho</th>
      <th>normalized_beta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Breezer</td>
      <td>0</td>
      <td>5.620958</td>
      <td>3.878248</td>
      <td>0.257848</td>
      <td>1.452272</td>
      <td>1.404630</td>
      <td>0.969142</td>
      <td>1.031841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Commencal</td>
      <td>0</td>
      <td>5.765467</td>
      <td>3.971145</td>
      <td>0.251817</td>
      <td>1.452311</td>
      <td>1.441604</td>
      <td>0.992950</td>
      <td>1.007101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IronHorse</td>
      <td>0</td>
      <td>5.613436</td>
      <td>3.934638</td>
      <td>0.254153</td>
      <td>1.426263</td>
      <td>1.403254</td>
      <td>0.983586</td>
      <td>1.016688</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Scott</td>
      <td>0</td>
      <td>5.398303</td>
      <td>3.821598</td>
      <td>0.261671</td>
      <td>1.410603</td>
      <td>1.350040</td>
      <td>0.955728</td>
      <td>1.046323</td>
    </tr>
  </tbody>
</table>
</div>



#### Similarity from function

If the similarity matrix file doesn't fit in RAM and we don't have a precomputed similarity matrix, we can define a custom similarity function to generate the similarity matrix on the fly. If the matrix is large or the similarity function is computationally expensive, then this approach can take a long time.


```python
def euclidean_similarity(a, b):
    euclidean_distance = norm(a - b)
    return 1 / (1 + euclidean_distance)
```


```python
penguin_metacommunity = make_metacommunity(
    penguin_counts, 
    species_order=species_order, 
    similarity_function=euclidean_similarity, 
    features=penguin_features
)
```


```python
penguin_metacommunity.subcommunities_to_dataframe(viewpoint=0)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>community</th>
      <th>viewpoint</th>
      <th>alpha</th>
      <th>rho</th>
      <th>beta</th>
      <th>gamma</th>
      <th>normalized_alpha</th>
      <th>normalized_rho</th>
      <th>normalized_beta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Biscoe</td>
      <td>0</td>
      <td>4.068632</td>
      <td>1.635146</td>
      <td>0.611566</td>
      <td>2.638069</td>
      <td>1.997771</td>
      <td>0.802886</td>
      <td>1.245507</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dream</td>
      <td>0</td>
      <td>5.253772</td>
      <td>1.735989</td>
      <td>0.576041</td>
      <td>3.539491</td>
      <td>1.934772</td>
      <td>0.639301</td>
      <td>1.564208</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Torgersen</td>
      <td>0</td>
      <td>7.106383</td>
      <td>3.155993</td>
      <td>0.316858</td>
      <td>2.251711</td>
      <td>1.000000</td>
      <td>0.444107</td>
      <td>2.251711</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

### Command line interface

For calculating diversity indices from the command line, we will need the following files, which can be downloaded from this [Gist](https://gist.github.com/Elliot-D-Hill/1200c74c83a4be1f63f9ef56b333bb99).

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

Once the files are downloaded, execute
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

In addition to being sensitive to frequency, it often makes sense to account for similarity in a diversity measure. For example, a community of 2 different types of rodents, may be considered less diverse as the same community where one of the rodent types was replaced by the same number of individuals of a bird species. [Reeve et al.](https://arxiv.org/abs/1404.6520) and [Leinster and Cobbold](https://doi.org/10.1890/10-2402.1) present a general mathematically rigorous way of incorporating similarity measures into Hill's framework. The result is a family of similarity-sensitive diversity indices parameterized by the same viewpoint parameter as well as the similarity function used for the species in the meta- or subcommunities of interest. As opposed to accounting for distinct species and their frequency, these similarity-sensitive diversity measures can be interpreted as accounting for different possibly overlapping clusters of mutually similar species and their combined frequencies.

### One package to calculate them all

The `diversity` package is able to calculate all of the similarity- and frequency-sensitive subcommunity and metacommunity diversity measures described in [Reeve et al.](https://arxiv.org/abs/1404.6520). See the paper for more in-depth information on their derivation and interpretation.

## Alternatives

To date, we know of no other python package that implements the similarity-sensitive diversity measures calculated by `diversity`. An [R package](https://github.com/boydorr/rdiversity) and a [julia package](https://github.com/EcoJulia/Diversity.jl) exist. However, both packages require the species similarities to be stored in the form of a matrix in memory. That approach does not scale to the amount of species in some applications, such as immune repertoires. `diversity` allows the user to store the similarity matrix in a file, or simply provide a python function that computes the similarities on the fly.
