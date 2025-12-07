from load_constants import load_constants
from nist import *

def process_sequence(seq_name: str, sequence: str, results: list, block_size: int, expect_pi: list) -> None:
    """Обрабатывает одну последовательность и добавляет результаты в список."""
    results.append((f"Frequency Bitwise Test ({seq_name})", freq_bitwise_test(sequence)))
    results.append((f"Same Bits Test ({seq_name})", same_bits_test(sequence)))
    results.append((f"Longest Test ({seq_name})", longest_test(sequence, block_size, expect_pi)))

def print_ones_count(sequences: dict) -> None:
    """Выводит количество единиц в последовательностях."""
    for name, seq in sequences.items():
        print(f"Count of '1' in {name}: {seq.count('1')}")

def main():
    constants = load_constants('constants.json')
    
    BLOCK_SIZE = constants['BLOCK_SIZE']
    EXPECT_PI = constants['EXPECT_PI']
    
    sequences = {
        'first C++ sequence': constants['CPP_SEQ'],
        'second C++ sequence': constants['CPP_SEQ2'],
        'first Java sequence': constants['JAVA_SEQ'],
        'second Java sequence': constants['JAVA_SEQ2']
    }
    
    results1 = []
    process_sequence("C++", constants['CPP_SEQ'], results1, BLOCK_SIZE, EXPECT_PI)
    process_sequence("Java", constants['JAVA_SEQ'], results1, BLOCK_SIZE, EXPECT_PI)
    write_results(results1, constants['RESULT_FILE'])
    
    results2 = []
    process_sequence("C++", constants['CPP_SEQ2'], results2, BLOCK_SIZE, EXPECT_PI)
    process_sequence("Java", constants['JAVA_SEQ2'], results2, BLOCK_SIZE, EXPECT_PI)
    write_results(results2, constants['RESULT_FILE2'])
    
    print_ones_count(sequences)

if __name__ == "__main__":
    main()