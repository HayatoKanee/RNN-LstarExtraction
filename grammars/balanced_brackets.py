"""
Balanced Brackets - A context-free language that is NOT regular.

This language cannot be recognized by any DFA, but RNNs can learn it well.
It will be interesting to see what happens when we try to extract a DFA.
"""

import random
from typing import Tuple, List, Set


def is_balanced_brackets(s: str) -> bool:
    """
    Check if a string has balanced parentheses.
    
    Rules:
    - '(' must have matching ')'
    - ')' cannot appear before its matching '('
    - Empty string is balanced
    
    This is the classic context-free language that requires a stack/counter.
    """
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count < 0:  # More closing than opening
                return False
        else:
            return False  # Invalid character
    
    return count == 0  # All brackets matched


def generate_balanced_brackets(max_depth: int = 3) -> str:
    """Generate a random balanced bracket string."""
    if max_depth == 0:
        return ""
    
    s = ""
    depth = 0
    
    while depth > 0 or (depth == 0 and random.random() < 0.7 and max_depth > 0):
        if depth >= max_depth:
            # Must close
            s += ")"
            depth -= 1
        elif depth == 0:
            # Must open
            s += "("
            depth += 1
        else:
            # Can open or close
            if random.random() < 0.6:  # Slight bias toward opening
                s += "("
                depth += 1
                max_depth -= 1
            else:
                s += ")"
                depth -= 1
    
    # Close any remaining open brackets
    while depth > 0:
        s += ")"
        depth -= 1
    
    return s


def generate_unbalanced_brackets(length: int) -> str:
    """Generate a random unbalanced bracket string."""
    s = ''.join(random.choice(['(', ')']) for _ in range(length))
    
    # Ensure it's actually unbalanced
    while is_balanced_brackets(s):
        # Perturb it
        if s and random.random() < 0.5:
            # Remove a character
            idx = random.randint(0, len(s) - 1)
            s = s[:idx] + s[idx+1:]
        else:
            # Add a character
            idx = random.randint(0, len(s))
            s = s[:idx] + random.choice(['(', ')']) + s[idx:]
    
    return s


def generate_dataset(num_samples: int = 5000, max_length: int = 20) -> List[Tuple[str, bool]]:
    """
    Generate a dataset of balanced and unbalanced bracket strings.
    
    Returns list of (string, is_balanced) tuples.
    """
    dataset = []
    
    # Always include edge cases
    dataset.append(("", True))  # Empty string is balanced
    dataset.append(("(", False))
    dataset.append((")", False))
    dataset.append(("()", True))
    dataset.append(("((", False))
    dataset.append(("))", False))
    dataset.append((")(", False))
    
    # Generate balanced examples
    balanced_count = num_samples // 2
    for _ in range(balanced_count):
        max_depth = random.randint(1, max_length // 2)
        s = generate_balanced_brackets(max_depth)
        if len(s) <= max_length:
            dataset.append((s, True))
    
    # Generate unbalanced examples
    unbalanced_count = num_samples - len(dataset)
    for _ in range(unbalanced_count):
        length = random.randint(1, max_length)
        s = generate_unbalanced_brackets(length)
        if len(s) <= max_length:
            dataset.append((s, False))
    
    # Remove duplicates
    dataset = list(set(dataset))
    
    # Shuffle
    random.shuffle(dataset)
    
    return dataset


def get_bracket_info() -> Tuple:
    """Return language info for consistency with Tomita grammars."""
    return is_balanced_brackets, "Balanced Brackets"


# Other non-regular languages that RNNs can learn:

def is_equal_ab(s: str) -> bool:
    """
    Language: {a^n b^n | n >= 0}
    Strings with equal number of a's followed by equal number of b's.
    """
    if not s:
        return True
    
    # Find transition point
    a_count = 0
    for i, char in enumerate(s):
        if char == 'a':
            a_count += 1
        elif char == 'b':
            # Rest must all be b's
            b_count = len(s) - i
            return a_count == b_count and all(c == 'b' for c in s[i:])
        else:
            return False
    
    # All a's, no b's
    return False


def is_palindrome(s: str) -> bool:
    """
    Binary palindromes over {0,1}.
    Another classic context-free but not regular language.
    """
    return s == s[::-1]


def is_arithmetic_expr(s: str) -> bool:
    """
    Simple arithmetic expressions with matched parentheses.
    Alphabet: {a, +, *, (, )}
    Grammar: E -> a | E+E | E*E | (E)
    """
    # This is a simplified check - full parsing would be more complex
    if not s:
        return False
    
    # Check parentheses balance
    paren_count = 0
    prev_op = True  # Expect operand at start
    
    for i, char in enumerate(s):
        if char == '(':
            paren_count += 1
            prev_op = True  # Expect operand after (
        elif char == ')':
            paren_count -= 1
            if paren_count < 0:
                return False
            prev_op = False  # Can have operator after )
        elif char == 'a':
            if not prev_op:
                return False  # Two operands in a row
            prev_op = False
        elif char in '+*':
            if prev_op:
                return False  # Two operators in a row
            prev_op = True
        else:
            return False  # Invalid character
    
    # Must end with operand or ), and parentheses balanced
    return paren_count == 0 and not prev_op


def is_copy_language(s: str) -> bool:
    """
    Language: {ww | w in {0,1}*}
    Strings that are repetitions of themselves.
    """
    if len(s) % 2 != 0:
        return False
    
    mid = len(s) // 2
    return s[:mid] == s[mid:]


# Summary of non-regular languages for testing
NON_REGULAR_LANGUAGES = {
    'balanced_brackets': (is_balanced_brackets, "Balanced parentheses"),
    'equal_ab': (is_equal_ab, "a^n b^n"),
    'palindrome': (is_palindrome, "Binary palindromes"),
    'arithmetic': (is_arithmetic_expr, "Simple arithmetic expressions"),
    'copy': (is_copy_language, "ww repetition language")
}