import pytest
import diversity
from numpy import isclose
from run import process_input_file


TEST_QS = [0, 0.5, 1, 1.5, 2, 3, 4, 100, 1000]
TEST_QDS = 3


@pytest.fixture
def mock_input_file(tmp_path):
    mock_csv_data = [
        "name_1,1,0,AAA",
        "name_2,1,0,BBB",
        "name_3,1,0,CCC",
    ]
    datafile = tmp_path / "input.csv"
    datafile.write_text("\n".join(mock_csv_data))
    return str(datafile)


@pytest.fixture
def mock_matrix_file(tmp_path):
    mock_csv_data = [
        "1.0,0.0,0.0",
        "0.0,1.0,0.0",
        "0.0,0.0,1.0",
    ]
    datafile = tmp_path / "temp.csv"
    datafile.write_text("\n".join(mock_csv_data))
    return str(datafile)


def test_alpha_from_file(mock_input_file, mock_matrix_file):
    features, counts = process_input_file(mock_input_file)
    for q in TEST_QS:
        qDs = diversity.alpha(features, counts, q, filepath=mock_matrix_file)
        assert isclose(TEST_QDS, qDs)


def test_alpha_from_similarity_fn(mock_input_file):
    features, counts = process_input_file(mock_input_file)
    for q in TEST_QS:
        qDs = diversity.alpha(
            features, counts, q, similarity_fn=diversity.sequence_similarity)
        assert isclose(TEST_QDS, qDs)
