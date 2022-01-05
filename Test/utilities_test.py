"""Tests for Chubacabra.utilities."""
from copy import deepcopy

from numpy import array
from pytest import mark

from Chubacabra.utilities import register, UniqueRowsCorrespondence


UNIQUE_ROWS_CORRESPONDENCE_TEST_CASES = [
    {
        'description': 'Multiple columns; default nonunique key column.',
        'init_kwargs': {
            'data': array([[0, 1, 1],
                           [1, 0, 2],
                           [2, 1, 3],
                           [1, 0, 4]]),
            },
        'unique_keys': set([0,1,2]),
    }, {
        'description': 'Multiple columns; nondefault unique key column.',
        'init_kwargs': {
            'data': array([[0, 1, 1],
                           [1, 0, 2],
                           [2, 1, 3],
                           [1, 0, 4]]),
            'key_column_pos': 2
            },
        'unique_keys': set([1,2,3,4]),
    }, {
        'description': 'Single column; default nonunique key column.',
        'init_kwargs': {
            'data': array([['foo'],
                           ['bar'],
                           ['foo'],
                           ['baz'],
                           ['bap']]),
            },
        'unique_keys': set(['foo', 'bar', 'baz', 'bap']),
    }, {
        'description': 'Single row; default unique key column.',
        'init_kwargs': {
            'data': array([[0,5,1]])
            },
        'unique_keys': set([0]),
    }]
class TestUniqueRowsCorrespondence:
    """Tests Chubacabra.utilities.UniqueRowsCorrespondence."""

    def get_data_key_column(self, init_kwargs):
        """Returns the data and key_column_pos values tested."""
        data = init_kwargs['data']
        if 'key_column_pos' in init_kwargs:
            key_column_pos = init_kwargs['key_column_pos']
        else:
            key_column_pos = 0
        return (data, key_column_pos)

    @mark.parametrize('test_case', UNIQUE_ROWS_CORRESPONDENCE_TEST_CASES)
    def test_unique_row_index(self, test_case):
        """Tests .unique_row_index indexes expected unique keys."""
        data, key_column_pos = deepcopy(
            self.get_data_key_column(test_case['init_kwargs']))
        correspondence = UniqueRowsCorrespondence(**test_case['init_kwargs'])
        row_index = correspondence.unique_row_index
        indexed_keys = set(data[row_index, key_column_pos])
        assert len(row_index) == len(test_case['unique_keys'])
        assert indexed_keys == test_case['unique_keys']

    @mark.parametrize('test_case', UNIQUE_ROWS_CORRESPONDENCE_TEST_CASES)
    def test_unique_keys(self, test_case):
        """Tests .unique_keys returns expected unique keys."""
        correspondence = UniqueRowsCorrespondence(**test_case['init_kwargs'])
        keys = correspondence.unique_keys
        assert len(keys) == len(test_case['unique_keys'])
        assert set(keys) == test_case['unique_keys']

    @mark.parametrize('test_case', UNIQUE_ROWS_CORRESPONDENCE_TEST_CASES)
    def test_unique_row_index_keys_ordering(self, test_case):
        """Tests .unique_row_index and .unique_keys orderings agree."""
        data, key_column_pos = deepcopy(
            self.get_data_key_column(test_case['init_kwargs']))
        correspondence = UniqueRowsCorrespondence(**test_case['init_kwargs'])
        row_index = correspondence.unique_row_index
        indexed_keys = data[row_index, key_column_pos]
        keys = correspondence.unique_keys
        assert all(indexed_keys == keys)

    @mark.parametrize('test_case', UNIQUE_ROWS_CORRESPONDENCE_TEST_CASES)
    def test_key_to_unique_pos(self, test_case):
        """Tests .key_to_unique_pos and .unique_keys orderings agree."""
        correspondence = UniqueRowsCorrespondence(**test_case['init_kwargs'])
        keys = correspondence.unique_keys
        key_to_unique_pos = correspondence.key_to_unique_pos
        assert set(key_to_unique_pos.keys()) == test_case['unique_keys']
        for pos, key in enumerate(keys):
            assert key_to_unique_pos[key] == pos, f'keys: {keys}; key: {key}; pos: {pos}'

    @mark.parametrize('test_case', UNIQUE_ROWS_CORRESPONDENCE_TEST_CASES)
    def test_row_to_unique_pos(self, test_case):
        """Tests .row_to_unique_pos and .unique_keys orderings agree."""
        data, key_column_pos = deepcopy(
            self.get_data_key_column(test_case['init_kwargs']))
        correspondence = UniqueRowsCorrespondence(**test_case['init_kwargs'])
        ordered_keys = list(correspondence.unique_keys)
        row_to_unique_pos = correspondence.row_to_unique_pos
        assert row_to_unique_pos.shape == (data.shape[0],)
        for row, key in enumerate(data[:,key_column_pos]):
            expected_pos = ordered_keys.index(key)
            assert row_to_unique_pos[row] == expected_pos, f'row: {row}; key: {key}; ordered_keys: {ordered_keys}; expected_pos: {expected_pos}'


REGISTER_TEST_CASES = [
    {
        'description': 'Item not yet in registry.',
        'kwargs': {
            'item': 'foo',
            'registry': {'bar': 15}},
        'return_value': 1,
        'modified_registry': {'bar': 15, 'foo': 1},
    }, {
        'description': 'Item already in registry.',
        'kwargs': {
            'item': 2022,
            'registry': {'bar': 15, 'foo': 100, 2022: 'foo'}},
        'return_value': 'foo',
        'modified_registry': {'bar': 15, 'foo': 100, 2022: 'foo'},
    }]
class TestRegister:
    """Tests Chubacabra.utilities.register."""

    @mark.parametrize('test_case', REGISTER_TEST_CASES)
    def test_cases(self, test_case):
        actual_return_value = register(**test_case['kwargs'])
        assert actual_return_value == test_case['return_value']
        assert test_case['kwargs']['registry'] == test_case['modified_registry']
