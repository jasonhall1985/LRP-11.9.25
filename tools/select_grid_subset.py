#!/usr/bin/env python3
"""
GRID Subset Selection for ICU Pretraining
=========================================

Selects optimal GRID word subset based on viseme similarity to ICU phrases.
Uses the viseme mapping system to find GRID words with similar visual patterns
to ICU target phrases for effective pretraining.

Strategy:
1. Calculate viseme similarity between each ICU phrase and all GRID words
2. Select top-K most similar words per ICU class
3. Ensure balanced representation across speakers and word types
4. Generate training manifest for GRID pretraining phase

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import argparse

# Add utils to path for viseme mapper
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from utils.viseme_mapper import VisemeMapper
except ImportError:
    print("❌ Could not import VisemeMapper. Please ensure utils/viseme_mapper.py exists.")
    sys.exit(1)

class GRIDSubsetSelector:
    """Selects optimal GRID subset for ICU pretraining based on viseme similarity."""
    
    def __init__(self, grid_manifest_path: Path, viseme_mapper: VisemeMapper = None):
        self.grid_manifest_path = Path(grid_manifest_path)
        self.viseme_mapper = viseme_mapper or VisemeMapper()
        
        # ICU target phrases
        self.icu_phrases = {
            'doctor': 'doctor',
            'my_mouth_is_dry': 'my mouth is dry',
            'i_need_to_move': 'i need to move', 
            'pillow': 'pillow'
        }
        
        # Load GRID manifests
        self.sentence_manifest = None
        self.word_manifest = None
        self.load_manifests()
    
    def load_manifests(self):
        """Load GRID sentence and word manifests."""
        sentence_path = self.grid_manifest_path.parent / "grid_sentence_manifest.csv"
        word_path = self.grid_manifest_path.parent / "grid_word_manifest.csv"
        
        if sentence_path.exists():
            self.sentence_manifest = pd.read_csv(sentence_path)
            print(f"📋 Loaded sentence manifest: {len(self.sentence_manifest)} entries")
        else:
            print(f"⚠️  Sentence manifest not found: {sentence_path}")
        
        if word_path.exists():
            self.word_manifest = pd.read_csv(word_path)
            print(f"📋 Loaded word manifest: {len(self.word_manifest)} entries")
        else:
            print(f"⚠️  Word manifest not found: {word_path}")
    
    def calculate_word_similarities(self) -> Dict[str, Dict[str, float]]:
        """Calculate viseme similarity between ICU phrases and all GRID words."""
        if self.word_manifest is None:
            print("❌ Word manifest not available")
            return {}
        
        print("🧮 Calculating viseme similarities...")
        
        # Get unique GRID words
        unique_words = self.word_manifest['word'].unique()
        print(f"📊 Found {len(unique_words)} unique GRID words")
        
        similarities = {}
        
        for icu_class, icu_phrase in self.icu_phrases.items():
            print(f"  Processing ICU phrase: '{icu_phrase}'")
            
            word_similarities = []
            for grid_word in unique_words:
                similarity = self.viseme_mapper.calculate_text_similarity(icu_phrase, grid_word)
                word_similarities.append((grid_word, similarity))
            
            # Sort by similarity (descending)
            word_similarities.sort(key=lambda x: x[1], reverse=True)
            similarities[icu_class] = dict(word_similarities)
            
            # Show top 10 most similar words
            print(f"    Top 10 similar words:")
            for word, sim in word_similarities[:10]:
                print(f"      {word}: {sim:.3f}")
        
        return similarities
    
    def select_balanced_subset(self, similarities: Dict[str, Dict[str, float]], 
                              words_per_class: int = 20, 
                              min_examples_per_word: int = 10) -> Dict[str, List[str]]:
        """
        Select balanced subset of GRID words for each ICU class.
        
        Args:
            similarities: Word similarity scores per ICU class
            words_per_class: Number of words to select per ICU class
            min_examples_per_word: Minimum examples required per word
        """
        print(f"🎯 Selecting balanced subset: {words_per_class} words per class")
        
        selected_words = {}
        
        for icu_class, word_sims in similarities.items():
            print(f"\n📝 Selecting words for '{icu_class}':")
            
            # Get words with sufficient examples
            valid_words = []
            for word, similarity in word_sims.items():
                word_count = len(self.word_manifest[self.word_manifest['word'] == word])
                if word_count >= min_examples_per_word:
                    valid_words.append((word, similarity, word_count))
            
            print(f"  Found {len(valid_words)} words with ≥{min_examples_per_word} examples")
            
            # Sort by similarity and select top words
            valid_words.sort(key=lambda x: x[1], reverse=True)
            selected = valid_words[:words_per_class]
            
            selected_words[icu_class] = [word for word, _, _ in selected]
            
            print(f"  Selected {len(selected)} words:")
            for word, similarity, count in selected:
                print(f"    {word}: {similarity:.3f} ({count} examples)")
        
        return selected_words
    
    def build_pretraining_manifest(self, selected_words: Dict[str, List[str]], 
                                  max_examples_per_word: int = 50) -> pd.DataFrame:
        """Build training manifest for GRID pretraining."""
        print(f"📋 Building pretraining manifest...")
        
        training_entries = []
        
        for icu_class, words in selected_words.items():
            print(f"\n📊 Processing class '{icu_class}' ({len(words)} words):")
            
            class_total = 0
            for word in words:
                # Get all examples of this word
                word_examples = self.word_manifest[self.word_manifest['word'] == word].copy()
                
                # Limit examples per word to prevent imbalance
                if len(word_examples) > max_examples_per_word:
                    word_examples = word_examples.sample(n=max_examples_per_word, random_state=42)
                
                # Add ICU class label
                word_examples['icu_class'] = icu_class
                word_examples['icu_phrase'] = self.icu_phrases[icu_class]
                
                training_entries.append(word_examples)
                class_total += len(word_examples)
                
                print(f"  {word}: {len(word_examples)} examples")
            
            print(f"  Class total: {class_total} examples")
        
        # Combine all entries
        training_manifest = pd.concat(training_entries, ignore_index=True)
        
        print(f"\n✅ Pretraining manifest created: {len(training_manifest)} total examples")
        
        # Show class distribution
        class_counts = training_manifest['icu_class'].value_counts()
        print("📈 Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} examples")
        
        return training_manifest
    
    def analyze_speaker_distribution(self, manifest: pd.DataFrame) -> Dict:
        """Analyze speaker distribution in the selected subset."""
        speaker_stats = {}
        
        # Overall speaker distribution
        speaker_counts = manifest['speaker_id'].value_counts()
        speaker_stats['overall'] = speaker_counts.to_dict()
        
        # Speaker distribution per class
        speaker_stats['by_class'] = {}
        for icu_class in manifest['icu_class'].unique():
            class_data = manifest[manifest['icu_class'] == icu_class]
            class_speakers = class_data['speaker_id'].value_counts()
            speaker_stats['by_class'][icu_class] = class_speakers.to_dict()
        
        return speaker_stats
    
    def save_pretraining_subset(self, manifest: pd.DataFrame, output_dir: Path):
        """Save pretraining subset and analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training manifest
        manifest_path = output_dir / "grid_pretraining_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"💾 Saved pretraining manifest: {manifest_path}")
        
        # Save word selection summary
        summary_path = output_dir / "grid_subset_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("GRID PRETRAINING SUBSET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Examples: {len(manifest)}\n")
            f.write(f"Unique Words: {manifest['word'].nunique()}\n")
            f.write(f"Speakers: {manifest['speaker_id'].nunique()}\n")
            f.write(f"ICU Classes: {manifest['icu_class'].nunique()}\n\n")
            
            # Class distribution
            f.write("CLASS DISTRIBUTION:\n")
            class_counts = manifest['icu_class'].value_counts()
            for class_name, count in class_counts.items():
                f.write(f"  {class_name}: {count} examples\n")
            f.write("\n")
            
            # Word distribution per class
            f.write("WORDS PER CLASS:\n")
            for icu_class in manifest['icu_class'].unique():
                class_data = manifest[manifest['icu_class'] == icu_class]
                word_counts = class_data['word'].value_counts()
                f.write(f"{icu_class.upper()}:\n")
                for word, count in word_counts.items():
                    f.write(f"  {word}: {count} examples\n")
                f.write("\n")
            
            # Speaker distribution
            f.write("SPEAKER DISTRIBUTION:\n")
            speaker_counts = manifest['speaker_id'].value_counts()
            for speaker, count in speaker_counts.items():
                f.write(f"  {speaker}: {count} examples\n")
        
        print(f"📊 Saved subset summary: {summary_path}")
        
        # Save class-specific manifests
        for icu_class in manifest['icu_class'].unique():
            class_data = manifest[manifest['icu_class'] == icu_class]
            class_path = output_dir / f"grid_{icu_class}_manifest.csv"
            class_data.to_csv(class_path, index=False)
            print(f"💾 Saved {icu_class} manifest: {class_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Select GRID Subset for ICU Pretraining')
    parser.add_argument('--manifest-dir', default='manifests',
                       help='Directory containing GRID manifests')
    parser.add_argument('--output-dir', default='manifests/pretraining',
                       help='Output directory for pretraining subset')
    parser.add_argument('--words-per-class', type=int, default=20,
                       help='Number of words to select per ICU class')
    parser.add_argument('--max-examples-per-word', type=int, default=50,
                       help='Maximum examples per word to prevent imbalance')
    parser.add_argument('--min-examples-per-word', type=int, default=10,
                       help='Minimum examples required per word')
    
    args = parser.parse_args()
    
    print("🎯 GRID SUBSET SELECTION FOR ICU PRETRAINING")
    print("=" * 50)
    
    # Check manifest directory
    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.exists():
        print(f"❌ Manifest directory not found: {manifest_dir}")
        print("Please run build_grid_manifest.py first")
        return 1
    
    # Initialize selector
    selector = GRIDSubsetSelector(manifest_dir / "grid_word_manifest.csv")
    
    if selector.word_manifest is None:
        print("❌ Could not load GRID manifests")
        return 1
    
    # Calculate similarities
    similarities = selector.calculate_word_similarities()
    
    if not similarities:
        print("❌ Could not calculate similarities")
        return 1
    
    # Select balanced subset
    selected_words = selector.select_balanced_subset(
        similarities, 
        words_per_class=args.words_per_class,
        min_examples_per_word=args.min_examples_per_word
    )
    
    # Build pretraining manifest
    pretraining_manifest = selector.build_pretraining_manifest(
        selected_words,
        max_examples_per_word=args.max_examples_per_word
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    selector.save_pretraining_subset(pretraining_manifest, output_dir)
    
    print(f"\n✅ GRID subset selection complete")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Ready for GRID pretraining phase")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
