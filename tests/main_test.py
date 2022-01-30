"""Tests for diversity.__main__."""
from argparse import Namespace

from numpy import inf
from pytest import mark

from diversity.__main__ import main

MAIN_TEST_CASES = [
    {
        "description": "disjoint communities; uniform counts; uniform inter-community similarities; viewpoint 0.",
        "args": Namespace(
            input_filepath="counts.tsv",
            output_filepath="diversities.tsv",
            similarity_matrix_filepath="similarities.tsv",
            viewpoint=[0],
            log_level="WARNING",
            subcommunity_column="subcommunity",
            species_column="species",
            count_column="count",
            chunk_size=1,
            subcommunities=None,
        ),
        "input_filecontents": (
            "subcommunity\tspecies\tcount\n"
            "subcommunity_1\tspecies_1\t1\n"
            "subcommunity_1\tspecies_2\t1\n"
            "subcommunity_1\tspecies_3\t1\n"
            "subcommunity_2\tspecies_4\t1\n"
            "subcommunity_2\tspecies_5\t1\n"
            "subcommunity_2\tspecies_6\t1\n"
        ),
        "similarities_filecontents": (
            "species_1\tspecies_2\tspecies_3\tspecies_4\tspecies_5\tspecies_6\n"
            "1.0\t0.5\t0.5\t0.7\t0.7\t0.7\n"
            "0.5\t1.0\t0.5\t0.7\t0.7\t0.7\n"
            "0.5\t0.5\t1.0\t0.7\t0.7\t0.7\n"
            "0.7\t0.7\t0.7\t1.0\t0.5\t0.5\n"
            "0.7\t0.7\t0.7\t0.5\t1.0\t0.5\n"
            "0.7\t0.7\t0.7\t0.5\t0.5\t1.0\n"
        ),
        "output_filecontents": (
            "community\tviewpoint\talpha\trho\tbeta\tgamma\tnormalized_alpha\tnormalized_rho\tnormalized_beta\n"
            "subcommunity_1\t0.00\t3.0000\t2.0500\t0.4878\t1.4634\t1.5000\t1.0250\t0.9756\n"
            "subcommunity_2\t0.00\t3.0000\t2.0500\t0.4878\t1.4634\t1.5000\t1.0250\t0.9756\n"
            "metacommunity\t0.00\t3.0000\t2.0500\t0.4878\t1.4634\t1.5000\t1.0250\t0.9756\n"
        ),
    },
    {
        "description": "overlapping communities; non-uniform counts; non-uniform inter-community similarities; viewpoint 2.",
        "args": Namespace(
            input_filepath="foo_counts.tsv",
            output_filepath="bar_counts.tsv",
            similarity_matrix_filepath="baz_similarities.tsv",
            viewpoint=[2, 101, 102, inf],
            log_level="WARNING",
            subcommunity_column="subcommunity",
            species_column="species",
            count_column="count",
            chunk_size=1,
            subcommunities=None,
        ),
        "input_filecontents": (
            "species\tsubcommunity\tcount\n"
            "species_1\tsubcommunity_1\t2\n"
            "species_2\tsubcommunity_1\t3\n"
            "species_1\tsubcommunity_2\t5\n"
            "species_3\tsubcommunity_2\t1\n"
        ),
        "similarities_filecontents": (
            "species_1\tspecies_2\tspecies_3\n"
            "1.0\t0.5\t0.1\n"
            "0.5\t1.0\t0.2\n"
            "0.1\t0.2\t1.0\n"
        ),
        "output_filecontents": (
            "community\tviewpoint\talpha\trho\tbeta\tgamma\tnormalized_alpha\tnormalized_rho\tnormalized_beta\n"
            "subcommunity_1\t2.00\t2.8947\t1.9194\t0.5210\t1.4745\t1.3158\t0.8724\t1.1462\n"
            "subcommunity_2\t2.00\t2.4444\t1.6587\t0.6029\t1.4570\t1.3333\t0.9047\t1.1053\n"
            "metacommunity\t2.00\t2.6304\t1.7678\t0.5627\t1.4649\t1.3253\t0.8898\t1.1235\n"
            "subcommunity_1\t101.00\t2.7641\t1.6836\t0.5940\t1.2908\t1.2564\t0.7653\t1.3067\n"  # 2.76408362, 1.68357822, 0.593973, 1.29084362, 1.25640164, 0.76526283, 1.30674059
            "subcommunity_2\t101.00\t2.1608\t1.5610\t0.6406\t1.2814\t1.1786\t0.8515\t1.1744\n"  # 2.16079876, 1.56104879, 0.64059497, 1.28140391, 1.1786175, 0.85148116, 1.17442411
            "metacommunity\t101.00\t2.1739\t1.5705\t0.5987\t1.2849\t1.1858\t0.7713\t1.1816\n"  # 2.1739359070171167, 1.5705327595028187, 0.5986709767832536, 1.2848640573127161, 1.185766665953564, 0.7713202342679308, 1.1815641039291136
            "subcommunity_1\t102.00\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\n"  # 2.75, 1.675, 0.59701493, 1.27906977, 1.25, 0.76136364, 1.31343284
            "subcommunity_2\t102.00\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\n"  # 2.15686275, 1.53333333, 0.65217391, 1.27906977, 1.17647059, 0.83636364, 1.19565217
            "metacommunity\t102.00\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\n"  # 2.156862745098039, 1.5333333333333337, 0.5970149253731344, 1.2790697674418605, 1.176470588235294, 0.7613636363636362, 1.1956521739130435
            "subcommunity_1\tinf\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\n"  # 2.75, 1.675, 0.59701493, 1.27906977, 1.25, 0.76136364, 1.31343284
            "subcommunity_2\tinf\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\n"  # 2.15686275, 1.53333333, 0.65217391, 1.27906977, 1.17647059, 0.83636364, 1.19565217
            "metacommunity\tinf\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\n"  # 2.156862745098039, 1.5333333333333337, 0.5970149253731344, 1.2790697674418605, 1.176470588235294, 0.7613636363636362, 1.1956521739130435
        ),
    },
    {
        "description": "test column naming.",
        "args": Namespace(
            input_filepath="foo_counts.tsv",
            output_filepath="bar_counts.tsv",
            similarity_matrix_filepath="baz_similarities.tsv",
            viewpoint=[2, 101, 102, inf],
            log_level="WARNING",
            subcommunity_column="subcommunity_",
            species_column="species_",
            count_column="count_",
            chunk_size=1,
            subcommunities=None,
        ),
        "input_filecontents": (
            "species\tsubcommunity_\tcount_\tspecies_\tsubcommunity\tcount\n"
            "species_5\tsubcommunity_1\t2\tspecies_1\tsubcommunity_x\t100\n"
            "species_9\tsubcommunity_1\t3\tspecies_2\tsubcommunity_y\t55\n"
            "species_9\tsubcommunity_2\t5\tspecies_1\tsubcommunity_z\t12\n"
            "species_0\tsubcommunity_2\t1\tspecies_3\tsubcommunity_z\t19\n"
        ),
        "similarities_filecontents": (
            "species_1\tspecies_2\tspecies_3\n"
            "1.0\t0.5\t0.1\n"
            "0.5\t1.0\t0.2\n"
            "0.1\t0.2\t1.0\n"
        ),
        "output_filecontents": (
            "community\tviewpoint\talpha\trho\tbeta\tgamma\tnormalized_alpha\tnormalized_rho\tnormalized_beta\n"
            "subcommunity_1\t2.00\t2.8947\t1.9194\t0.5210\t1.4745\t1.3158\t0.8724\t1.1462\n"
            "subcommunity_2\t2.00\t2.4444\t1.6587\t0.6029\t1.4570\t1.3333\t0.9047\t1.1053\n"
            "metacommunity\t2.00\t2.6304\t1.7678\t0.5627\t1.4649\t1.3253\t0.8898\t1.1235\n"
            "subcommunity_1\t101.00\t2.7641\t1.6836\t0.5940\t1.2908\t1.2564\t0.7653\t1.3067\n"  # 2.76408362, 1.68357822, 0.593973, 1.29084362, 1.25640164, 0.76526283, 1.30674059
            "subcommunity_2\t101.00\t2.1608\t1.5610\t0.6406\t1.2814\t1.1786\t0.8515\t1.1744\n"  # 2.16079876, 1.56104879, 0.64059497, 1.28140391, 1.1786175, 0.85148116, 1.17442411
            "metacommunity\t101.00\t2.1739\t1.5705\t0.5987\t1.2849\t1.1858\t0.7713\t1.1816\n"  # 2.1739359070171167, 1.5705327595028187, 0.5986709767832536, 1.2848640573127161, 1.185766665953564, 0.7713202342679308, 1.1815641039291136
            "subcommunity_1\t102.00\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\n"  # 2.75, 1.675, 0.59701493, 1.27906977, 1.25, 0.76136364, 1.31343284
            "subcommunity_2\t102.00\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\n"  # 2.15686275, 1.53333333, 0.65217391, 1.27906977, 1.17647059, 0.83636364, 1.19565217
            "metacommunity\t102.00\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\n"  # 2.156862745098039, 1.5333333333333337, 0.5970149253731344, 1.2790697674418605, 1.176470588235294, 0.7613636363636362, 1.1956521739130435
            "subcommunity_1\tinf\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\n"  # 2.75, 1.675, 0.59701493, 1.27906977, 1.25, 0.76136364, 1.31343284
            "subcommunity_2\tinf\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\n"  # 2.15686275, 1.53333333, 0.65217391, 1.27906977, 1.17647059, 0.83636364, 1.19565217
            "metacommunity\tinf\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\n"  # 2.156862745098039, 1.5333333333333337, 0.5970149253731344, 1.2790697674418605, 1.176470588235294, 0.7613636363636362, 1.1956521739130435
        ),
    },
    {
        "description": "Test chunk_size.",
        "args": Namespace(
            input_filepath="foo_counts.tsv",
            output_filepath="bar_counts.tsv",
            similarity_matrix_filepath="baz_similarities.tsv",
            viewpoint=[2, 101, 102, inf],
            log_level="WARNING",
            subcommunity_column="subcommunity",
            species_column="species",
            count_column="count",
            chunk_size=2,
            subcommunities=None,
        ),
        "input_filecontents": (
            "species\tsubcommunity\tcount\n"
            "species_1\tsubcommunity_1\t2\n"
            "species_2\tsubcommunity_1\t3\n"
            "species_1\tsubcommunity_2\t5\n"
            "species_3\tsubcommunity_2\t1\n"
        ),
        "similarities_filecontents": (
            "species_1\tspecies_2\tspecies_3\n"
            "1.0\t0.5\t0.1\n"
            "0.5\t1.0\t0.2\n"
            "0.1\t0.2\t1.0\n"
        ),
        "output_filecontents": (
            "community\tviewpoint\talpha\trho\tbeta\tgamma\tnormalized_alpha\tnormalized_rho\tnormalized_beta\n"
            "subcommunity_1\t2.00\t2.8947\t1.9194\t0.5210\t1.4745\t1.3158\t0.8724\t1.1462\n"
            "subcommunity_2\t2.00\t2.4444\t1.6587\t0.6029\t1.4570\t1.3333\t0.9047\t1.1053\n"
            "metacommunity\t2.00\t2.6304\t1.7678\t0.5627\t1.4649\t1.3253\t0.8898\t1.1235\n"
            "subcommunity_1\t101.00\t2.7641\t1.6836\t0.5940\t1.2908\t1.2564\t0.7653\t1.3067\n"  # 2.76408362, 1.68357822, 0.593973, 1.29084362, 1.25640164, 0.76526283, 1.30674059
            "subcommunity_2\t101.00\t2.1608\t1.5610\t0.6406\t1.2814\t1.1786\t0.8515\t1.1744\n"  # 2.16079876, 1.56104879, 0.64059497, 1.28140391, 1.1786175, 0.85148116, 1.17442411
            "metacommunity\t101.00\t2.1739\t1.5705\t0.5987\t1.2849\t1.1858\t0.7713\t1.1816\n"  # 2.1739359070171167, 1.5705327595028187, 0.5986709767832536, 1.2848640573127161, 1.185766665953564, 0.7713202342679308, 1.1815641039291136
            "subcommunity_1\t102.00\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\n"  # 2.75, 1.675, 0.59701493, 1.27906977, 1.25, 0.76136364, 1.31343284
            "subcommunity_2\t102.00\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\n"  # 2.15686275, 1.53333333, 0.65217391, 1.27906977, 1.17647059, 0.83636364, 1.19565217
            "metacommunity\t102.00\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\n"  # 2.156862745098039, 1.5333333333333337, 0.5970149253731344, 1.2790697674418605, 1.176470588235294, 0.7613636363636362, 1.1956521739130435
            "subcommunity_1\tinf\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\n"  # 2.75, 1.675, 0.59701493, 1.27906977, 1.25, 0.76136364, 1.31343284
            "subcommunity_2\tinf\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\n"  # 2.15686275, 1.53333333, 0.65217391, 1.27906977, 1.17647059, 0.83636364, 1.19565217
            "metacommunity\tinf\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\n"  # 2.156862745098039, 1.5333333333333337, 0.5970149253731344, 1.2790697674418605, 1.176470588235294, 0.7613636363636362, 1.1956521739130435
        ),
    },
    {
        "description": "overlapping communities; non-uniform counts; non-uniform inter-community similarities; viewpoint 2.",
        "args": Namespace(
            input_filepath="foo_counts.tsv",
            output_filepath="bar_counts.tsv",
            similarity_matrix_filepath="baz_similarities.tsv",
            viewpoint=[2, 101, 102, inf],
            log_level="WARNING",
            subcommunity_column="subcommunity",
            species_column="species",
            count_column="count",
            chunk_size=1,
            subcommunities={"subcommunity_1", "subcommunity_2"},
        ),
        "input_filecontents": (
            "species\tsubcommunity\tcount\n"
            "species_1\tsubcommunity_1\t2\n"
            "species_2\tsubcommunity_1\t3\n"
            "species_1\tsubcommunity_2\t5\n"
            "species_3\tsubcommunity_2\t1\n"
            "species_1\tsubcommunity_3\t3\n"
            "species_2\tsubcommunity_3\t5\n"
        ),
        "similarities_filecontents": (
            "species_1\tspecies_2\tspecies_3\n"
            "1.0\t0.5\t0.1\n"
            "0.5\t1.0\t0.2\n"
            "0.1\t0.2\t1.0\n"
        ),
        "output_filecontents": (
            "community\tviewpoint\talpha\trho\tbeta\tgamma\tnormalized_alpha\tnormalized_rho\tnormalized_beta\n"
            "subcommunity_1\t2.00\t2.8947\t1.9194\t0.5210\t1.4745\t1.3158\t0.8724\t1.1462\n"
            "subcommunity_2\t2.00\t2.4444\t1.6587\t0.6029\t1.4570\t1.3333\t0.9047\t1.1053\n"
            "metacommunity\t2.00\t2.6304\t1.7678\t0.5627\t1.4649\t1.3253\t0.8898\t1.1235\n"
            "subcommunity_1\t101.00\t2.7641\t1.6836\t0.5940\t1.2908\t1.2564\t0.7653\t1.3067\n"  # 2.76408362, 1.68357822, 0.593973, 1.29084362, 1.25640164, 0.76526283, 1.30674059
            "subcommunity_2\t101.00\t2.1608\t1.5610\t0.6406\t1.2814\t1.1786\t0.8515\t1.1744\n"  # 2.16079876, 1.56104879, 0.64059497, 1.28140391, 1.1786175, 0.85148116, 1.17442411
            "metacommunity\t101.00\t2.1739\t1.5705\t0.5987\t1.2849\t1.1858\t0.7713\t1.1816\n"  # 2.1739359070171167, 1.5705327595028187, 0.5986709767832536, 1.2848640573127161, 1.185766665953564, 0.7713202342679308, 1.1815641039291136
            "subcommunity_1\t102.00\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\n"  # 2.75, 1.675, 0.59701493, 1.27906977, 1.25, 0.76136364, 1.31343284
            "subcommunity_2\t102.00\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\n"  # 2.15686275, 1.53333333, 0.65217391, 1.27906977, 1.17647059, 0.83636364, 1.19565217
            "metacommunity\t102.00\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\n"  # 2.156862745098039, 1.5333333333333337, 0.5970149253731344, 1.2790697674418605, 1.176470588235294, 0.7613636363636362, 1.1956521739130435
            "subcommunity_1\tinf\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\n"  # 2.75, 1.675, 0.59701493, 1.27906977, 1.25, 0.76136364, 1.31343284
            "subcommunity_2\tinf\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\n"  # 2.15686275, 1.53333333, 0.65217391, 1.27906977, 1.17647059, 0.83636364, 1.19565217
            "metacommunity\tinf\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\n"  # 2.156862745098039, 1.5333333333333337, 0.5970149253731344, 1.2790697674418605, 1.176470588235294, 0.7613636363636362, 1.1956521739130435
        ),
    },
]


class TestMain:
    """Tests __main__.main."""

    def write_file(self, path, contents):
        """Writes contents into file at path."""
        with open(path, "w") as file:
            file.write(contents)

    @mark.parametrize("test_case", MAIN_TEST_CASES)
    def test_main(self, test_case, tmp_path):
        """Tests __main__.main."""
        test_case[
            "args"
        ].input_filepath = f"{tmp_path}/{test_case['args'].input_filepath}"
        test_case[
            "args"
        ].similarity_matrix_filepath = (
            f"{tmp_path}/{test_case['args'].similarity_matrix_filepath}"
        )
        test_case[
            "args"
        ].output_filepath = f"{tmp_path}/{test_case['args'].output_filepath}"

        self.write_file(
            test_case["args"].input_filepath,
            test_case["input_filecontents"],
        )
        self.write_file(
            test_case["args"].similarity_matrix_filepath,
            test_case["similarities_filecontents"],
        )

        main(test_case["args"])
        with open(test_case["args"].output_filepath, "r") as file:
            output_filecontents = file.read()
        assert output_filecontents == test_case["output_filecontents"]
