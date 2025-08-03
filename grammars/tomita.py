"""
Tomita Grammars Implementation
Based on Tomita (1982) - classic benchmark grammars for RNN learning
"""

from typing import Callable, Tuple, List
import re


def tomita_1(word: str) -> bool:
    """
    Tomita Grammar 1: 1*
    Accepts strings containing only 1s (no 0s allowed).
    """
    return not "0" in word


def tomita_2(word: str) -> bool:
    """
    Tomita Grammar 2: (10)*
    Accepts strings that are repetitions of "10".
    """
    return word == "10" * (int(len(word)/2))


# Not tomita 3: words containing an odd series of consecutive ones and then later an odd series of consecutive zeros
_not_tomita_3 = re.compile("((0|1)*0)*1(11)*(0(0|1)*1)*0(00)*(1(0|1)*)*$") 

def tomita_3(w: str) -> bool:
    """
    Tomita Grammar 3: Complement of specific pattern
    Accepts strings that are NOT:
    - words containing an odd series of consecutive ones and then later an odd series of consecutive zeros
    """
    return None is _not_tomita_3.match(w)  # complement of _not_tomita_3


def tomita_4(word: str) -> bool:
    """
    Tomita Grammar 4: No three consecutive 0s
    Accepts strings that don't contain "000".
    """
    return not "000" in word


def tomita_5(word: str) -> bool:
    """
    Tomita Grammar 5: Even 0s and even 1s
    Accepts strings with even count of both 0s and 1s.
    """
    return (word.count("0") % 2 == 0) and (word.count("1") % 2 == 0)


def tomita_6(word: str) -> bool:
    """
    Tomita Grammar 6: Difference of 0s and 1s divisible by 3
    Accepts strings where (#0s - #1s) mod 3 = 0.
    """
    return ((word.count("0") - word.count("1")) % 3) == 0


def tomita_7(word: str) -> bool:
    """
    Tomita Grammar 7: At most one occurrence of "10"
    Accepts strings with at most one occurrence of the substring "10".
    """
    return word.count("10") <= 1


# Dictionary of all Tomita grammars
TOMITA_GRAMMARS = {
    1: (tomita_1, "1* (no zeros allowed)"),
    2: (tomita_2, "(10)* (alternating 10 pattern)"),
    3: (tomita_3, "complement of odd consecutive 1s then odd consecutive 0s"),
    4: (tomita_4, "no three consecutive 0s"),
    5: (tomita_5, "even 0s AND even 1s"),
    6: (tomita_6, "(#0s - #1s) mod 3 = 0"),
    7: (tomita_7, "at most one occurrence of '10'")
}


def get_tomita_grammar(grammar_id: int) -> Tuple[Callable[[str], bool], str]:
    """
    Get Tomita grammar function and description by ID.
    
    Args:
        grammar_id: Grammar ID (1-7)
        
    Returns:
        Tuple of (grammar_function, description)
    """
    if grammar_id not in TOMITA_GRAMMARS:
        raise ValueError(f"Unknown Tomita grammar ID: {grammar_id}. Valid IDs are 1-7.")
    return TOMITA_GRAMMARS[grammar_id]


def generate_binary_strings(max_length: int) -> List[str]:
    """
    Generate all binary strings up to max_length.
    
    Args:
        max_length: Maximum string length
        
    Returns:
        List of binary strings including empty string
    """
    strings = ['']  # Include empty string
    for length in range(1, max_length + 1):
        for i in range(2**length):
            binary = bin(i)[2:].zfill(length)
            strings.append(binary)
    return strings


def test_grammars():
    """Test all Tomita grammars with example strings."""
    test_strings = ['', '0', '1', '00', '01', '10', '11', '000', '001', '010', 
                    '011', '100', '101', '110', '111', '1010', '0101', '1111']
    
    print("Testing Tomita Grammars")
    print("=" * 60)
    
    for grammar_id, (grammar_func, description) in TOMITA_GRAMMARS.items():
        print(f"\nTomita {grammar_id}: {description}")
        accepted = [s for s in test_strings if grammar_func(s)]
        print(f"Accepted: {accepted[:10]}{'...' if len(accepted) > 10 else ''}")
        print(f"Accept rate: {len(accepted)}/{len(test_strings)}")


if __name__ == "__main__":
    test_grammars()