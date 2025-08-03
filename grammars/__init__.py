"""Grammar definitions for UPCA-L* experiments."""

from .tomita import (
    tomita_1, tomita_2, tomita_3, tomita_4, 
    tomita_5, tomita_6, tomita_7,
    TOMITA_GRAMMARS,
    get_tomita_grammar,
    generate_binary_strings
)

__all__ = [
    'tomita_1', 'tomita_2', 'tomita_3', 'tomita_4',
    'tomita_5', 'tomita_6', 'tomita_7',
    'TOMITA_GRAMMARS',
    'get_tomita_grammar',
    'generate_binary_strings'
]