#!/usr/bin/env python3
"""
Viseme Mapping System for Three-Stage Training Pipeline
======================================================

Maps phonemes to visemes and calculates similarity between ICU phrases and GRID words
for optimal pretraining subset selection.

Visemes are visual speech units that group phonemes with similar lip/mouth movements.
This enables matching ICU phrases like "my mouth is dry" with GRID words that have
similar visual patterns.

Author: Augment Agent
Date: 2025-09-27
"""

import re
from typing import Dict, List, Tuple, Set
from collections import Counter
import numpy as np

# Phoneme to Viseme mapping based on visual similarity
# Using the 14-viseme system commonly used in lip-reading research
PHONEME_TO_VISEME = {
    # Bilabial sounds (lips together/close)
    'P': 'V1', 'B': 'V1', 'M': 'V1',
    
    # Labiodental sounds (lip-teeth contact)
    'F': 'V2', 'V': 'V2',
    
    # Dental/Alveolar sounds (tongue-teeth/alveolar ridge)
    'TH': 'V3', 'DH': 'V3',
    'T': 'V4', 'D': 'V4', 'S': 'V4', 'Z': 'V4', 'N': 'V4', 'L': 'V4',
    
    # Post-alveolar sounds
    'SH': 'V5', 'ZH': 'V5', 'CH': 'V5', 'JH': 'V5', 'R': 'V5',
    
    # Velar sounds (back of tongue)
    'K': 'V6', 'G': 'V6', 'NG': 'V6',
    
    # Vowels grouped by mouth shape
    'IY': 'V7', 'IH': 'V7', 'EY': 'V7',  # High front vowels (spread lips)
    'EH': 'V8', 'AE': 'V8',              # Mid-low front vowels
    'AA': 'V9', 'AO': 'V9', 'AH': 'V9',  # Low/central vowels (open mouth)
    'UH': 'V10', 'UW': 'V10', 'OW': 'V10', 'OY': 'V10',  # Back vowels (rounded lips)
    'AW': 'V11', 'AY': 'V11',            # Diphthongs
    'ER': 'V12', 'AX': 'V12',            # R-colored/schwa
    'Y': 'V13', 'W': 'V13', 'HH': 'V13', # Glides/fricatives
    'SILENCE': 'V14'                      # Silence/pause
}

# Common phoneme variations and aliases
PHONEME_ALIASES = {
    'TH0': 'TH', 'TH1': 'DH',
    'S0': 'S', 'S1': 'Z',
    'T0': 'T', 'T1': 'D',
    'F0': 'F', 'F1': 'V',
    'P0': 'P', 'P1': 'B',
    'K0': 'K', 'K1': 'G',
    'SIL': 'SILENCE', 'SP': 'SILENCE'
}

class VisemeMapper:
    """Maps phonemes to visemes and calculates visual similarity between words."""
    
    def __init__(self):
        self.phoneme_to_viseme = PHONEME_TO_VISEME.copy()
        self.phoneme_aliases = PHONEME_ALIASES.copy()
        
        # Add aliases to main mapping
        for alias, canonical in self.phoneme_aliases.items():
            if canonical in self.phoneme_to_viseme:
                self.phoneme_to_viseme[alias] = self.phoneme_to_viseme[canonical]
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phonemes using simple rule-based approach.
        For production use, integrate with a proper G2P library like g2p_en.
        """
        # Simple rule-based phoneme conversion for common ICU phrases
        text = text.lower().strip()
        
        # Dictionary of common words to phonemes (simplified)
        word_to_phonemes = {
            'doctor': ['D', 'AA', 'K', 'T', 'ER'],
            'my': ['M', 'AY'],
            'mouth': ['M', 'AW', 'TH'],
            'is': ['IH', 'Z'],
            'dry': ['D', 'R', 'AY'],
            'i': ['AY'],
            'need': ['N', 'IY', 'D'],
            'to': ['T', 'UW'],
            'move': ['M', 'UW', 'V'],
            'pillow': ['P', 'IH', 'L', 'OW'],
            'help': ['HH', 'EH', 'L', 'P'],
            'phone': ['F', 'OW', 'N'],
            'glasses': ['G', 'L', 'AE', 'S', 'IH', 'Z'],
            'water': ['W', 'AO', 'T', 'ER'],
            'pain': ['P', 'EY', 'N'],
            'nurse': ['N', 'ER', 'S'],
            'bed': ['B', 'EH', 'D'],
            'cold': ['K', 'OW', 'L', 'D'],
            'hot': ['HH', 'AA', 'T'],
            'yes': ['Y', 'EH', 'S'],
            'no': ['N', 'OW'],
            'please': ['P', 'L', 'IY', 'Z'],
            'thank': ['TH', 'AE', 'NG', 'K'],
            'you': ['Y', 'UW']
        }
        
        # Split text into words and convert each
        words = re.findall(r'\b\w+\b', text)
        all_phonemes = []
        
        for word in words:
            if word in word_to_phonemes:
                all_phonemes.extend(word_to_phonemes[word])
            else:
                # Fallback: simple letter-to-phoneme mapping
                for char in word:
                    if char in 'aeiou':
                        all_phonemes.append('AH')  # Generic vowel
                    else:
                        all_phonemes.append(char.upper())  # Use letter as phoneme
        
        return all_phonemes
    
    def phonemes_to_visemes(self, phonemes: List[str]) -> List[str]:
        """Convert phoneme sequence to viseme sequence."""
        visemes = []
        for phoneme in phonemes:
            # Handle phoneme aliases
            canonical_phoneme = self.phoneme_aliases.get(phoneme, phoneme)
            # Map to viseme
            viseme = self.phoneme_to_viseme.get(canonical_phoneme, 'V14')  # Default to silence
            visemes.append(viseme)
        return visemes
    
    def text_to_visemes(self, text: str) -> List[str]:
        """Convert text directly to viseme sequence."""
        phonemes = self.text_to_phonemes(text)
        return self.phonemes_to_visemes(phonemes)
    
    def calculate_viseme_similarity(self, visemes1: List[str], visemes2: List[str]) -> float:
        """
        Calculate similarity between two viseme sequences.
        Uses normalized edit distance and viseme frequency overlap.
        """
        if not visemes1 or not visemes2:
            return 0.0
        
        # Method 1: Normalized edit distance
        edit_distance = self._edit_distance(visemes1, visemes2)
        max_len = max(len(visemes1), len(visemes2))
        edit_similarity = 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0
        
        # Method 2: Viseme frequency overlap (Jaccard similarity)
        counter1 = Counter(visemes1)
        counter2 = Counter(visemes2)
        
        # Calculate intersection and union
        intersection = sum((counter1 & counter2).values())
        union = sum((counter1 | counter2).values())
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Method 3: Longest common subsequence
        lcs_length = self._longest_common_subsequence(visemes1, visemes2)
        lcs_similarity = (2 * lcs_length) / (len(visemes1) + len(visemes2))
        
        # Weighted combination of similarity measures
        combined_similarity = (
            0.4 * edit_similarity +
            0.3 * jaccard_similarity +
            0.3 * lcs_similarity
        )
        
        return combined_similarity
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate visual similarity between two text strings."""
        visemes1 = self.text_to_visemes(text1)
        visemes2 = self.text_to_visemes(text2)
        return self.calculate_viseme_similarity(visemes1, visemes2)
    
    def _edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate edit distance between two sequences."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def get_top_similar_words(self, target_phrase: str, candidate_words: List[str], 
                             top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find top-K most visually similar words to target phrase.
        Returns list of (word, similarity_score) tuples sorted by similarity.
        """
        similarities = []
        target_visemes = self.text_to_visemes(target_phrase)
        
        for word in candidate_words:
            word_visemes = self.text_to_visemes(word)
            similarity = self.calculate_viseme_similarity(target_visemes, word_visemes)
            similarities.append((word, similarity))
        
        # Sort by similarity (descending) and return top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def analyze_viseme_distribution(self, text: str) -> Dict[str, int]:
        """Analyze viseme distribution in text."""
        visemes = self.text_to_visemes(text)
        return dict(Counter(visemes))

def main():
    """Test the viseme mapping system."""
    mapper = VisemeMapper()
    
    # Test ICU phrases
    icu_phrases = [
        "doctor",
        "my mouth is dry", 
        "i need to move",
        "pillow"
    ]
    
    # Test GRID-like words
    grid_words = [
        "doctor", "water", "phone", "help", "move", "dry", "mouth",
        "place", "green", "blue", "red", "white", "black", "now",
        "please", "soon", "again", "with", "by", "at", "in"
    ]
    
    print("🎯 VISEME MAPPING SYSTEM TEST")
    print("=" * 50)
    
    # Test phoneme and viseme conversion
    for phrase in icu_phrases:
        phonemes = mapper.text_to_phonemes(phrase)
        visemes = mapper.phonemes_to_visemes(phonemes)
        print(f"\nPhrase: '{phrase}'")
        print(f"Phonemes: {phonemes}")
        print(f"Visemes: {visemes}")
        
        # Find most similar GRID words
        similar_words = mapper.get_top_similar_words(phrase, grid_words, top_k=5)
        print(f"Top similar words:")
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.3f}")
    
    print(f"\n✅ Viseme mapping system test complete")

if __name__ == "__main__":
    main()
