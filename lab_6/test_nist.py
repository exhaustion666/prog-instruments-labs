import math
import pytest
from load_constants import load_constants
from nist import freq_bitwise_test, same_bits_test, longest_test, write_results
from unittest.mock import mock_open, patch

def test_freq_bitwise_test_basic():
    """
    Test the basic case for freq_bitwise_test function.
    
    :params: None
    :returns: None
    """
    seq = "11010"
    result = freq_bitwise_test(seq)
    assert isinstance(result, float)
    assert 0 <= result <= 1


@pytest.mark.parametrize("seq, expected_range", [
    ("0" * 10, (0.0, 1.0)),
    ("1" * 10, (0.0, 1.0)),
    ("01" * 10, (0.0, 1.0)),
    ("0011" * 5, (0.0, 1.0)), 
])
def test_freq_bitwise_test_parametrized(seq, expected_range):
    """
    Parametrized testing of various sequences for freq_bitwise_test.
    
    :params:
        seq (str): Test binary sequence
        expected_range (tuple): Expected range for p-value (min, max)
    
    :returns: None
    """
    result = freq_bitwise_test(seq)
    assert expected_range[0] <= result <= expected_range[1]
    assert isinstance(result, float)


def test_same_bits_test_edge_cases():
    """
    Test edge cases for same_bits_test function.
    
    :params: None
    :returns: None
    """
    seq = "1" * 100 
    result = same_bits_test(seq)
    assert result == 0.0

    seq = "01" * 50
    result = same_bits_test(seq)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_longest_test():
    """
    Test longest_test function with a known sequence.
    
    :params: None
    :returns: None
    """
    seq = "1100110011" * 8
    block_size = 5
    expect_pi = [0.25, 0.25, 0.25, 0.25]
    
    result = longest_test(seq, block_size, expect_pi)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_write_results_with_mock():
    """
    Test write_results function using mock file object.
    
    :params: None
    :returns: None
    """
    results = [
        ("Test 1", 0.12345),
        ("Test 2", 0.98765),
        ("Test 3", 0.00123),
    ]
    
    mock_file = mock_open()
    with patch("builtins.open", mock_file):
        write_results(results, "test_output.txt")
    
    mock_file.assert_called_once_with("test_output.txt", mode='w', encoding='utf-8')
    
    handle = mock_file()
    assert handle.write.call_count >= 3


def test_load_constants_with_mock():
    """
    Test load_constants function using mock JSON file.
    
    :params: None
    :returns: None
    """
    mock_data = {
        'CPP_SEQ': '101010',
        'CPP_SEQ2': '010101',
        'JAVA_SEQ': '110011',
        'JAVA_SEQ2': '001100',
        'BLOCK_SIZE': 8,
        'EXPECT_PI': [0.2148, 0.3672, 0.2305, 0.1875],
        'RESULT_FILE': 'results1.txt',
        'RESULT_FILE2': 'results2.txt'
    }
    
    with patch('json.load') as mock_json_load:
        mock_json_load.return_value = mock_data
        
        with patch('builtins.open', mock_open()) as mock_file:
            result = load_constants('dummy.json')
    
    assert isinstance(result, dict)
    assert result['BLOCK_SIZE'] == 8
    assert len(result['EXPECT_PI']) == 4
    assert result['CPP_SEQ'] == '101010'


@pytest.mark.parametrize("func, seq", [
    (freq_bitwise_test, "0101010101"),
    (same_bits_test, "1100110011"),
])

def test_nist_functions_integration(func, seq):
    """
    Integration testing of nist module functions.
    
    :params:
        func (function): Function to test (freq_bitwise_test or same_bits_test)
        seq (str): Test binary sequence
    
    :returns: None
    """
    result = func(seq)
    
    assert isinstance(result, float)
    assert 0 <= result <= 1
    
    short_seq = "01"
    try:
        func(short_seq)
    except Exception as e:
        pytest.fail(f"Function {func.__name__} failed on short sequence: {e}")


def test_same_bits_test_special_case():
    """
    Test special case in same_bits_test where result should be 0.0.
    
    :params: None
    :returns: None
    """
    seq = "0" * 100
    result = same_bits_test(seq)
    assert result == 0.0


def test_mathematical_edge_cases():
    """
    Test mathematical edge cases for different sequence lengths.
    
    :params: None
    :returns: None
    """
    for length in [1, 10, 100, 1000]:
        seq = "01" * (length // 2)
        if len(seq) < length:
            seq += "0"
        
        result1 = freq_bitwise_test(seq)
        assert not math.isnan(result1)
        assert not math.isinf(result1)

        result2 = same_bits_test(seq)
        assert result2 == 0.0 or (0 <= result2 <= 1)


def test_error_handling():
    """
    Test error handling and edge cases for nist functions.
    
    :params: None
    :returns: None
    """
    assert freq_bitwise_test("") == 0.0
    assert same_bits_test("") == 0.0
    assert same_bits_test("0") == 0.0
    assert same_bits_test("1") == 0.0
    
    result = freq_bitwise_test("0")
    assert isinstance(result, float)
    assert 0 <= result <= 1
    
    assert longest_test("", 8, [0.25, 0.25, 0.25, 0.25]) == 0.0
    assert longest_test("0101", 0, [0.25, 0.25, 0.25, 0.25]) == 0.0
