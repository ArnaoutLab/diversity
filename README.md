# diversity: similarity-sensitive diversity indices

![Tests](https://github.com/Elliot-D-Hill/diversity/actions/workflows/tests.yml/badge.svg)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

- [diversity: similarity-sensitive diversity indices](#diversity-similarity-sensitive-diversity-indices)
  - [About](#about)
    - [Alternatives](#alternatives)
  - [Installation](#installation)
  - [Usage and Examples](#usage-and-examples)
    - [Command line interface](#command-line-interface)
    - [Python](#python)
    - [Similarity from an array or dataframe](#similarity-from-an-array-or-dataframe)
    - [Similarity from file](#similarity-from-file)
  - [Background](#background)
    - [Diversity indices](#diversity-indices)
    - [Partitioned diversity](#partitioned-diversity)
    - [Frequency-sensitive diversity](#frequency-sensitive-diversity)
    - [Similarity-sensitive diversity](#similarity-sensitive-diversity)
    - [One package to calculate them all](#one-package-to-calculate-them-all)


## About

For a rigorous mathematical treatment of the diversity indices calculated by `diversity`, see [Reeve et al., 2014](https://arxiv.org/abs/1404.6520). A brief informal discussion can be found in the [background section](#background).


### Alternatives

Diversity can be defined in various ways and software calculating the various diversity measures exists. To date, we know of no other python package that implements the similarity-sensitive partitioned diversity measures defined by [Reeve at al.](https://arxiv.org/abs/1404.6520). An [R package](https://github.com/boydorr/rdiversity) and a [julia package](https://github.com/EcoJulia/Diversity.jl) exist. However, both packages require the species similarities to be stored in the form of a matrix in memory. That approach does not scale to the amount of species in some applications, such as immune repertoires. `diversity` allows the user to store the similarity matrix in a file, or simply provide a python function that computes the similarities on the fly.


## Installation

`diversity` requires python version 3.9 or higher.

To install, execute:
```bash
pip install diversity
```

**Dependencies (pip should install them automatically):**
- numpy >= 1.22.0
- pandas >= 1.3.5


To run unit tests, make sure you have `pytest` installed, clone the
repository, and execute from within the repository's directory:
```bash
pytest
```


## Usage and Examples


### Command line interface

`diversity` is a versatile python package that supports execution directly from the commandline as a module (via `python -m`).

The example below uses the `penguin_counts.csv` and `penguin_similarity_matrix.csv` files, which can be downloaded from this [gist](https://gist.github.com/Elliot-D-Hill/f6e9db0aebe561a363e8758c72a0acfc).

Once the files are downloaded, execute

```bash
python -m diversity -i penguin_counts.csv -s penguin_similarity_matrix.csv -v 0 1 inf
```

to obtain all subcommunity diversity measures for the viewpoint
parameter values 0, 1, and infinity:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align:  right;">
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
      <td>Biscoe</td>
      <td>0.00</td>
      <td>3.1490</td>
      <td>1.7035</td>
      <td>0.5870</td>
      <td>1.8929</td>
      <td>1.5462</td>
      <td>0.8365</td>
      <td>1.1955</td>
    </tr>
    <tr>
      <td>Dream</td>
      <td>0.00</td>
      <td>4.0194</td>
      <td>2.0671</td>
      <td>0.4838</td>
      <td>1.9939</td>
      <td>1.4802</td>
      <td>0.7612</td>
      <td>1.3137</td>
    </tr>
    <tr>
      <td>Torgersen</td>
      <td>0.00</td>
      <td>7.1064</td>
      <td>4.1783</td>
      <td>0.2393</td>
      <td>1.7008</td>
      <td>1.0000</td>
      <td>0.5880</td>
      <td>1.7008</td>
    </tr>
    <tr>
      <td>metacommunity</td>
      <td>0.00</td>
      <td>4.0264</td>
      <td>2.1856</td>
      <td>0.5001</td>
      <td>1.9031</td>
      <td>1.4451</td>
      <td>0.7738</td>
      <td>1.3101</td>
    </tr>
    <tr>
      <td>Biscoe</td>
      <td>1.00</td>
      <td>3.0279</td>
      <td>1.6028</td>
      <td>0.6239</td>
      <td>1.8891</td>
      <td>1.4867</td>
      <td>0.7870</td>
      <td>1.2707</td>
    </tr>
    <tr>
      <td>Dream</td>
      <td>1.00</td>
      <td>4.0142</td>
      <td>2.0314</td>
      <td>0.4923</td>
      <td>1.9761</td>
      <td>1.4783</td>
      <td>0.7481</td>
      <td>1.3367</td>
    </tr>
    <tr>
      <td>Torgersen</td>
      <td>1.00</td>
      <td>7.1064</td>
      <td>4.1783</td>
      <td>0.2393</td>
      <td>1.7008</td>
      <td>1.0000</td>
      <td>0.5880</td>
      <td>1.7008</td>
    </tr>
    <tr>
      <td>metacommunity</td>
      <td>1.00</td>
      <td>3.7877</td>
      <td>2.0014</td>
      <td>0.4997</td>
      <td>1.8925</td>
      <td>1.4031</td>
      <td>0.7414</td>
      <td>1.3488</td>
    </tr>
    <tr>
      <td>Biscoe</td>
      <td>inf</td>
      <td>2.5730</td>
      <td>1.3105</td>
      <td>0.7631</td>
      <td>1.7008</td>
      <td>1.2634</td>
      <td>0.6435</td>
      <td>1.5540</td>
    </tr>
    <tr>
      <td>Dream</td>
      <td>inf</td>
      <td>3.8344</td>
      <td>1.7187</td>
      <td>0.5818</td>
      <td>1.7008</td>
      <td>1.4121</td>
      <td>0.6329</td>
      <td>1.5799</td>
    </tr>
    <tr>
      <td>Torgersen</td>
      <td>inf</td>
      <td>7.1064</td>
      <td>4.1783</td>
      <td>0.2393</td>
      <td>1.7008</td>
      <td>1.0000</td>
      <td>0.5880</td>
      <td>1.7008</td>
    </tr>
    <tr>
      <td>metacommunity</td>
      <td>inf</td>
      <td>2.5730</td>
      <td>1.3105</td>
      <td>0.2393</td>
      <td>1.7008</td>
      <td>1.0000</td>
      <td>0.5880</td>
      <td>1.5540</td>
    </tr>
  </tbody>
</table>
</div>

Notes:
- the input filepath (`-i`) and the similarity matrix filepath (`-s`)
  can be URLs to data files hosted on the web
- you can use .csv or .tsv for input files, but output is tab-delimited
- output can be piped (piping will not include log statements in the output)
- viewpoint parameter values of 100 or larger are treated like infinity
- if only some of the subcommunities should be included in the analysis,
  specify these using the `-b` option (for example: `-b Biscoe Torgersen`)
- for further options execute `python -m diversity -h`


### Python

```python
from diversity.metacommunity import make_metacommunity
from pandas import read_csv
from numpy.linalg import norm
```

The data for the following examples will be loaded from this [gist](https://gist.github.com/Elliot-D-Hill/f6e9db0aebe561a363e8758c72a0acfc)

```python
gist_url = 'https://gist.githubusercontent.com/Elliot-D-Hill/f6e9db0aebe561a363e8758c72a0acfc/raw/ef29ba813d2b0b8a09c245175e96c9828693f25d/'
```

### Similarity from an array or dataframe

First we load the [Palmer penguins dataset](https://github.com/allisonhorst/palmerpenguins). We have aggregated this dataset into subcommunity-by-species counts.

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

Next, we load a similiarty matrix as a pandas dataframe (but it could also be a numpy array).

```python
penguin_similarity_matrix_filepath = gist_url + 'penguin_similarity_matrix.csv'
penguin_similarity_matrix = read_csv(penguin_similarity_matrix_filepath)
```

Finally, we create a Metacommunity object.

```python
penguin_metacommunity = make_metacommunity(
    penguin_counts, 
    similarity_matrix=penguin_similarity_matrix
)
```

Metacommunty objects have convenience functions for calculating all diviersty measures at once for a given viewpoint.

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
      <td>3.149044</td>
      <td>1.703508</td>
      <td>0.587024</td>
      <td>1.892887</td>
      <td>1.546237</td>
      <td>0.836453</td>
      <td>1.195525</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dream</td>
      <td>0</td>
      <td>4.019417</td>
      <td>2.067072</td>
      <td>0.483776</td>
      <td>1.993902</td>
      <td>1.480205</td>
      <td>0.761227</td>
      <td>1.313668</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Torgersen</td>
      <td>0</td>
      <td>7.106383</td>
      <td>4.178339</td>
      <td>0.239330</td>
      <td>1.700768</td>
      <td>1.000000</td>
      <td>0.587970</td>
      <td>1.700768</td>
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
      <td>2.930631</td>
      <td>1.526693</td>
      <td>0.655011</td>
      <td>1.885246</td>
      <td>1.438992</td>
      <td>0.749634</td>
      <td>1.333985</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dream</td>
      <td>2</td>
      <td>4.009001</td>
      <td>1.997261</td>
      <td>0.500686</td>
      <td>1.958037</td>
      <td>1.476369</td>
      <td>0.735518</td>
      <td>1.359586</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Torgersen</td>
      <td>2</td>
      <td>7.106383</td>
      <td>4.178339</td>
      <td>0.239330</td>
      <td>1.700768</td>
      <td>1.000000</td>
      <td>0.587970</td>
      <td>1.700768</td>
    </tr>
  </tbody>
</table>
</div>

We can also compute all metacommunity measures at once. 

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
      <td>4.026442</td>
      <td>2.18565</td>
      <td>0.500075</td>
      <td>1.903052</td>
      <td>1.445054</td>
      <td>0.773784</td>
      <td>1.31013</td>
    </tr>
  </tbody>
</table>
</div>

We can also calculate diveristy measures individually for subcommunities or metacommunties.

```python
penguin_metacommunity.normalized_subcommunity_alpha(viewpoint=0)
```

    array([1.54623705, 1.48020455, 1.        ])


### Similarity from file

For large datasets, the similarity matrix may not fit in RAM. To avoid loading the entire matrix into RAM, use the similarity_matrix_filepath argument to read a file from a hard disk drive.


```python
penguin_metacommunity = make_metacommunity(
    penguin_counts, 
    similarity_matrix_filepath=penguin_similarity_matrix_filepath
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
      <td>3.149044</td>
      <td>1.703508</td>
      <td>0.587024</td>
      <td>1.892887</td>
      <td>1.546237</td>
      <td>0.836453</td>
      <td>1.195525</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dream</td>
      <td>0</td>
      <td>4.019417</td>
      <td>2.067072</td>
      <td>0.483776</td>
      <td>1.993902</td>
      <td>1.480205</td>
      <td>0.761227</td>
      <td>1.313668</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Torgersen</td>
      <td>0</td>
      <td>7.106383</td>
      <td>4.178339</td>
      <td>0.239330</td>
      <td>1.700768</td>
      <td>1.000000</td>
      <td>0.587970</td>
      <td>1.700768</td>
    </tr>
  </tbody>
</table>
</div>


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

