"""Tests for diversity.diversity."""
from pytest import fixture
from pandas import read_csv, DataFrame
from pandas.testing import assert_frame_equal

from diversity.metacommunity import Metacommunity

TEST_VIEWPOINT = 0
TEST_DF = DataFrame(
    {
        "community": ["subcommunity_1", "subcommunity_2"],
        "viewpoint": [0, 0],
        "alpha": [3.0, 3.0],
        "rho": [2.05, 2.05],
        "beta": [0.487805, 0.487805],
        "gamma": [1.463415, 1.463415],
        "normalized_alpha": [1.5, 1.5],
        "normalized_rho": [1.025, 1.025],
        "normalized_beta": [0.97561, 0.97561],
    }
)


@fixture
def mock_input_file(tmp_path):
    mock_csv_data = [
        "subcommunity,species,count",
        "subcommunity_1,species_1,1",
        "subcommunity_1,species_2,1",
        "subcommunity_1,species_3,1",
        "subcommunity_2,species_4,1",
        "subcommunity_2,species_5,1",
        "subcommunity_2,species_6,1",
    ]
    datafile = tmp_path / "input.csv"
    datafile.write_text("\n".join(mock_csv_data))
    return str(datafile)


@fixture
def mock_matrix_file(tmp_path):
    mock_csv_data = [
        "species_1,species_2,species_3,species_4,species_5,species_6",
        "1.0,0.5,0.5,0.7,0.7,0.7",
        "0.5,1.0,0.5,0.7,0.7,0.7",
        "0.5,0.5,1.0,0.7,0.7,0.7",
        "0.7,0.7,0.7,1.0,0.5,0.5",
        "0.7,0.7,0.7,0.5,1.0,0.5",
        "0.7,0.7,0.7,0.5,0.5,1.0",
    ]
    datafile = tmp_path / "matrix.csv"
    datafile.write_text("\n".join(mock_csv_data))
    return str(datafile)


@fixture
def mock_empty_matrix_file(tmp_path):
    datafile = tmp_path / "empty_matrix.csv"
    return str(datafile)


def test_subcommunity_from_file(mock_input_file, mock_matrix_file):
    df = read_csv(mock_input_file)
    meta = Metacommunity(df, similarities_filepath=mock_matrix_file)
    output_df = meta.subcommunities_to_dataframe(viewpoint=TEST_VIEWPOINT)
    assert_frame_equal(TEST_DF, output_df, check_exact=False)
