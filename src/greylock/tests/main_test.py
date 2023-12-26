"""Tests for diversity.__main__."""
from argparse import Namespace

from numpy import inf
from pytest import mark

from greylock.__main__ import main

MAIN_TEST_CASES = [
    {
        "description": "disjoint communities; uniform counts; uniform inter-community similarities; viewpoint 0.",
        "args": Namespace(
            input_filepath="counts.tsv",
            output_filepath="diversities.tsv",
            similarity="similarities.tsv",
            viewpoint=[0],
            log_level="WARNING",
            chunk_size=1,
        ),
        "input_filecontents": (
            "subcommunity_1\tsubcommunity_2\n"
            "1\t0\n"
            "1\t0\n"
            "1\t0\n"
            "0\t1\n"
            "0\t1\n"
            "0\t1\n"
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
            "community\tviewpoint\talpha\trho\tbeta\tgamma\tnormalized_alpha\tnormalized_rho\tnormalized_beta\trho_hat\tbeta_hat\n"
            "metacommunity\t0\t3.0000\t2.0500\t0.4878\t1.4634\t1.5000\t1.0250\t0.9756\t1.0500\t-0.0244\n"
            "subcommunity_1\t0\t3.0000\t2.0500\t0.4878\t1.4634\t1.5000\t1.0250\t0.9756\t1.0500\t-0.0244\n"
            "subcommunity_2\t0\t3.0000\t2.0500\t0.4878\t1.4634\t1.5000\t1.0250\t0.9756\t1.0500\t-0.0244\n"
        ),
    },
    {
        "description": "overlapping communities; non-uniform counts; non-uniform inter-community similarities; viewpoint 2.",
        "args": Namespace(
            input_filepath="foo_counts.tsv",
            output_filepath="bar_counts.tsv",
            similarity="baz_similarities.tsv",
            viewpoint=[2, 101, 102, inf],
            log_level="WARNING",
            chunk_size=1,
        ),
        "input_filecontents": (
            "subcommunity_1\tsubcommunity_2\n" "2\t5\n" "3\t0\n" "0\t1\n"
        ),
        "similarities_filecontents": (
            "species_1\tspecies_2\tspecies_3\n"
            "1.0\t0.5\t0.1\n"
            "0.5\t1.0\t0.2\n"
            "0.1\t0.2\t1.0\n"
        ),
        "output_filecontents": (
            "community\tviewpoint\talpha\trho\tbeta\tgamma\tnormalized_alpha\tnormalized_rho\tnormalized_beta\trho_hat\tbeta_hat\n"
            "metacommunity\t2.0000\t2.6304\t1.7678\t0.5627\t1.4649\t1.3253\t0.8898\t1.1235\t0.7562\t0.0742\n"
            "subcommunity_1\t2.0000\t2.8947\t1.9194\t0.5210\t1.4745\t1.3158\t0.8724\t1.1462\t0.9194\t0.0420\n"
            "subcommunity_2\t2.0000\t2.4444\t1.6587\t0.6029\t1.4570\t1.3333\t0.9047\t1.1053\t0.6587\t0.2058\n"
            "metacommunity\t101.0000\t2.1739\t1.5705\t0.5987\t1.2849\t1.1858\t0.7713\t1.1816\t0.5645\t0.1894\n"
            "subcommunity_1\t101.0000\t2.7641\t1.6836\t0.5940\t1.2908\t1.2564\t0.7653\t1.3067\t0.6836\t0.1879\n"
            "subcommunity_2\t101.0000\t2.1608\t1.5610\t0.6406\t1.2814\t1.1786\t0.8515\t1.1744\t0.5610\t0.2812\n"
            "metacommunity\t102.0000\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\t0.5333\t0.1940\n"
            "subcommunity_1\t102.0000\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\t0.6750\t0.1940\n"
            "subcommunity_2\t102.0000\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\t0.5333\t0.3043\n"
            "metacommunity\tinf\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\t0.5333\t0.1940\n"
            "subcommunity_1\tinf\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\t0.6750\t0.1940\n"
            "subcommunity_2\tinf\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\t0.5333\t0.3043\n"
        ),
    },
    {
        "description": "Test chunk_size.",
        "args": Namespace(
            input_filepath="foo_counts.tsv",
            output_filepath="bar_counts.tsv",
            similarity="baz_similarities.tsv",
            viewpoint=[2, 101, 102, inf],
            log_level="WARNING",
            chunk_size=2,
        ),
        "input_filecontents": (
            "subcommunity_1\tsubcommunity_2\n" "2\t5\n" "3\t0\n" "0\t1\n"
        ),
        "similarities_filecontents": (
            "species_1\tspecies_2\tspecies_3\n"
            "1.0\t0.5\t0.1\n"
            "0.5\t1.0\t0.2\n"
            "0.1\t0.2\t1.0\n"
        ),
        "output_filecontents": (
            "community\tviewpoint\talpha\trho\tbeta\tgamma\tnormalized_alpha\tnormalized_rho\tnormalized_beta\trho_hat\tbeta_hat\n"
            "metacommunity\t2.0000\t2.6304\t1.7678\t0.5627\t1.4649\t1.3253\t0.8898\t1.1235\t0.7562\t0.0742\n"
            "subcommunity_1\t2.0000\t2.8947\t1.9194\t0.5210\t1.4745\t1.3158\t0.8724\t1.1462\t0.9194\t0.0420\n"
            "subcommunity_2\t2.0000\t2.4444\t1.6587\t0.6029\t1.4570\t1.3333\t0.9047\t1.1053\t0.6587\t0.2058\n"
            "metacommunity\t101.0000\t2.1739\t1.5705\t0.5987\t1.2849\t1.1858\t0.7713\t1.1816\t0.5645\t0.1894\n"
            "subcommunity_1\t101.0000\t2.7641\t1.6836\t0.5940\t1.2908\t1.2564\t0.7653\t1.3067\t0.6836\t0.1879\n"
            "subcommunity_2\t101.0000\t2.1608\t1.5610\t0.6406\t1.2814\t1.1786\t0.8515\t1.1744\t0.5610\t0.2812\n"
            "metacommunity\t102.0000\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\t0.5333\t0.1940\n"
            "subcommunity_1\t102.0000\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\t0.6750\t0.1940\n"
            "subcommunity_2\t102.0000\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\t0.5333\t0.3043\n"
            "metacommunity\tinf\t2.1569\t1.5333\t0.5970\t1.2791\t1.1765\t0.7614\t1.1957\t0.5333\t0.1940\n"
            "subcommunity_1\tinf\t2.7500\t1.6750\t0.5970\t1.2791\t1.2500\t0.7614\t1.3134\t0.6750\t0.1940\n"
            "subcommunity_2\tinf\t2.1569\t1.5333\t0.6522\t1.2791\t1.1765\t0.8364\t1.1957\t0.5333\t0.3043\n"
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
        test_case["args"].similarity = f"{tmp_path}/{test_case['args'].similarity}"
        test_case[
            "args"
        ].output_filepath = f"{tmp_path}/{test_case['args'].output_filepath}"

        self.write_file(
            test_case["args"].input_filepath,
            test_case["input_filecontents"],
        )
        self.write_file(
            test_case["args"].similarity,
            test_case["similarities_filecontents"],
        )
        main(test_case["args"])
        with open(test_case["args"].output_filepath, "r") as file:
            output_filecontents = file.read()
        print(output_filecontents)
        print(test_case["output_filecontents"])
        assert output_filecontents == test_case["output_filecontents"]
