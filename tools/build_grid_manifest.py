#!/usr/bin/env python3
"""
GRID Corpus Manifest Builder
============================

Scans GRID corpus directory and builds comprehensive manifest for three-stage training.
Extracts speaker information, sentence structure, and word-level metadata for 
optimal subset selection based on viseme similarity to ICU phrases.

GRID Corpus Structure:
- Speakers: s1, s2, ..., s34
- Sentences: 6-word structure "command color preposition letter digit adverb"
- Example: "bin blue at a nine now"

Author: Augment Agent  
Date: 2025-09-27
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import argparse
from collections import defaultdict, Counter

class GRIDManifestBuilder:
    """Builds comprehensive manifest from GRID corpus directory."""
    
    def __init__(self, grid_root: Path):
        self.grid_root = Path(grid_root)
        self.manifest_data = []
        
        # GRID sentence structure (6-word format)
        self.word_positions = {
            0: 'command',    # bin, lay, place, set
            1: 'color',      # blue, green, red, white
            2: 'preposition', # at, by, in, with
            3: 'letter',     # a-z (excluding w)
            4: 'digit',      # 1-9, zero
            5: 'adverb'      # again, now, please, soon
        }
        
        # Expected GRID vocabulary
        self.grid_vocabulary = {
            'command': ['bin', 'lay', 'place', 'set'],
            'color': ['blue', 'green', 'red', 'white'],
            'preposition': ['at', 'by', 'in', 'with'],
            'letter': [chr(i) for i in range(ord('a'), ord('z')+1) if chr(i) != 'w'],
            'digit': ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
            'adverb': ['again', 'now', 'please', 'soon']
        }
    
    def scan_grid_directory(self) -> List[Dict]:
        """Scan GRID directory and extract all video/audio files with metadata."""
        
        if not self.grid_root.exists():
            print(f"❌ GRID directory not found: {self.grid_root}")
            return []
        
        print(f"📂 Scanning GRID corpus: {self.grid_root}")
        
        # Supported file extensions
        video_extensions = {'.mp4', '.avi', '.mov', '.mpg', '.mpeg'}
        audio_extensions = {'.wav', '.mp3', '.flac'}
        
        files_found = 0
        speakers_found = set()
        
        # Scan all speaker directories
        for speaker_dir in self.grid_root.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            # Check if this looks like a speaker directory (s1, s2, etc.)
            speaker_match = re.match(r's(\d+)', speaker_dir.name.lower())
            if not speaker_match:
                continue
                
            speaker_id = speaker_dir.name.lower()
            speaker_num = int(speaker_match.group(1))
            speakers_found.add(speaker_id)
            
            print(f"  📁 Processing speaker: {speaker_id}")
            
            # Scan files in speaker directory
            for file_path in speaker_dir.rglob('*'):
                if not file_path.is_file():
                    continue
                    
                file_ext = file_path.suffix.lower()
                if file_ext not in (video_extensions | audio_extensions):
                    continue
                
                # Extract sentence information from filename
                sentence_info = self.parse_grid_filename(file_path.name)
                if not sentence_info:
                    continue
                
                # Build manifest entry
                entry = {
                    'file_path': str(file_path),
                    'speaker_id': speaker_id,
                    'speaker_num': speaker_num,
                    'filename': file_path.name,
                    'file_type': 'video' if file_ext in video_extensions else 'audio',
                    'sentence_id': sentence_info['sentence_id'],
                    'sentence_text': sentence_info['sentence_text'],
                    'words': sentence_info['words'],
                    'command': sentence_info['words'][0] if len(sentence_info['words']) > 0 else '',
                    'color': sentence_info['words'][1] if len(sentence_info['words']) > 1 else '',
                    'preposition': sentence_info['words'][2] if len(sentence_info['words']) > 2 else '',
                    'letter': sentence_info['words'][3] if len(sentence_info['words']) > 3 else '',
                    'digit': sentence_info['words'][4] if len(sentence_info['words']) > 4 else '',
                    'adverb': sentence_info['words'][5] if len(sentence_info['words']) > 5 else '',
                    'word_count': len(sentence_info['words']),
                    'duration_estimate': self.estimate_duration(sentence_info['words'])
                }
                
                self.manifest_data.append(entry)
                files_found += 1
        
        print(f"✅ Found {files_found} files from {len(speakers_found)} speakers")
        print(f"📊 Speakers: {sorted(speakers_found)}")
        
        return self.manifest_data
    
    def parse_grid_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse GRID filename to extract sentence information.
        
        GRID filenames typically follow patterns like:
        - s1_swwv4s.mpg (sentence ID: swwv4s)
        - s1_lgir6n.wav (sentence ID: lgir6n)
        """
        # Remove extension
        name_without_ext = Path(filename).stem
        
        # Pattern: speaker_sentenceID
        match = re.match(r's(\d+)_([a-z0-9]+)', name_without_ext.lower())
        if not match:
            return None
        
        speaker_num = int(match.group(1))
        sentence_id = match.group(2)
        
        # Decode sentence ID to words
        words = self.decode_sentence_id(sentence_id)
        if not words:
            return None
        
        return {
            'sentence_id': sentence_id,
            'sentence_text': ' '.join(words),
            'words': words
        }
    
    def decode_sentence_id(self, sentence_id: str) -> Optional[List[str]]:
        """
        Decode GRID sentence ID to word sequence.
        
        GRID uses a compact encoding where each character/digit represents a word:
        - 1st char: command (b=bin, l=lay, p=place, s=set)
        - 2nd char: color (b=blue, g=green, r=red, w=white)  
        - 3rd char: preposition (a=at, b=by, i=in, w=with)
        - 4th char: letter (a-z except w)
        - 5th char: digit (0-9)
        - 6th char: adverb (a=again, n=now, p=please, s=soon)
        """
        if len(sentence_id) != 6:
            return None
        
        # Decoding mappings
        command_map = {'b': 'bin', 'l': 'lay', 'p': 'place', 's': 'set'}
        color_map = {'b': 'blue', 'g': 'green', 'r': 'red', 'w': 'white'}
        prep_map = {'a': 'at', 'b': 'by', 'i': 'in', 'w': 'with'}
        digit_map = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
        adverb_map = {'a': 'again', 'n': 'now', 'p': 'please', 's': 'soon'}
        
        try:
            words = [
                command_map[sentence_id[0]],
                color_map[sentence_id[1]], 
                prep_map[sentence_id[2]],
                sentence_id[3],  # letter as-is
                digit_map[sentence_id[4]],
                adverb_map[sentence_id[5]]
            ]
            return words
        except KeyError:
            # Invalid sentence ID
            return None
    
    def estimate_duration(self, words: List[str]) -> float:
        """Estimate video duration based on word count and average speaking rate."""
        # Average speaking rate: ~150 words per minute = 2.5 words per second
        # GRID sentences are typically 1-2 seconds
        base_duration = len(words) * 0.3  # 0.3 seconds per word
        return round(base_duration + 0.5, 1)  # Add 0.5s for pauses
    
    def build_word_level_manifest(self) -> pd.DataFrame:
        """Build word-level manifest for individual word extraction."""
        word_entries = []
        
        for entry in self.manifest_data:
            for i, word in enumerate(entry['words']):
                word_entry = {
                    'file_path': entry['file_path'],
                    'speaker_id': entry['speaker_id'],
                    'speaker_num': entry['speaker_num'],
                    'sentence_id': entry['sentence_id'],
                    'word': word,
                    'word_position': i,
                    'word_type': self.word_positions[i],
                    'start_time_estimate': i * 0.3,  # Rough estimate
                    'end_time_estimate': (i + 1) * 0.3,
                    'duration_estimate': 0.3
                }
                word_entries.append(word_entry)
        
        return pd.DataFrame(word_entries)
    
    def generate_statistics(self) -> Dict:
        """Generate comprehensive statistics about the GRID corpus."""
        if not self.manifest_data:
            return {}
        
        df = pd.DataFrame(self.manifest_data)
        
        stats = {
            'total_files': len(df),
            'total_speakers': df['speaker_id'].nunique(),
            'speakers': sorted(df['speaker_id'].unique()),
            'file_types': df['file_type'].value_counts().to_dict(),
            'total_sentences': df['sentence_id'].nunique(),
            'avg_files_per_speaker': len(df) / df['speaker_id'].nunique(),
            'word_statistics': {},
            'vocabulary_coverage': {}
        }
        
        # Word-level statistics
        all_words = []
        for words in df['words']:
            all_words.extend(words)
        
        word_counter = Counter(all_words)
        stats['word_statistics'] = {
            'total_words': len(all_words),
            'unique_words': len(word_counter),
            'most_common_words': word_counter.most_common(10),
            'word_frequency': dict(word_counter)
        }
        
        # Vocabulary coverage by category
        for category, expected_words in self.grid_vocabulary.items():
            category_words = df[category].value_counts().to_dict()
            coverage = len(set(category_words.keys()) & set(expected_words))
            stats['vocabulary_coverage'][category] = {
                'expected': len(expected_words),
                'found': len(category_words),
                'coverage': coverage / len(expected_words),
                'missing': list(set(expected_words) - set(category_words.keys()))
            }
        
        return stats
    
    def save_manifest(self, output_path: Path, include_word_level: bool = True):
        """Save manifest to CSV files."""
        if not self.manifest_data:
            print("❌ No manifest data to save")
            return
        
        # Save sentence-level manifest
        sentence_df = pd.DataFrame(self.manifest_data)
        sentence_path = output_path / "grid_sentence_manifest.csv"
        sentence_df.to_csv(sentence_path, index=False)
        print(f"💾 Saved sentence manifest: {sentence_path}")
        
        # Save word-level manifest
        if include_word_level:
            word_df = self.build_word_level_manifest()
            word_path = output_path / "grid_word_manifest.csv"
            word_df.to_csv(word_path, index=False)
            print(f"💾 Saved word manifest: {word_path}")
        
        # Save statistics
        stats = self.generate_statistics()
        stats_path = output_path / "grid_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("GRID CORPUS STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Files: {stats['total_files']}\n")
            f.write(f"Total Speakers: {stats['total_speakers']}\n")
            f.write(f"Speakers: {', '.join(stats['speakers'])}\n")
            f.write(f"File Types: {stats['file_types']}\n")
            f.write(f"Total Sentences: {stats['total_sentences']}\n")
            f.write(f"Avg Files per Speaker: {stats['avg_files_per_speaker']:.1f}\n\n")
            
            f.write("WORD STATISTICS:\n")
            f.write(f"Total Words: {stats['word_statistics']['total_words']}\n")
            f.write(f"Unique Words: {stats['word_statistics']['unique_words']}\n")
            f.write("Most Common Words:\n")
            for word, count in stats['word_statistics']['most_common_words']:
                f.write(f"  {word}: {count}\n")
            
            f.write("\nVOCABULARY COVERAGE:\n")
            for category, coverage in stats['vocabulary_coverage'].items():
                f.write(f"{category.upper()}:\n")
                f.write(f"  Expected: {coverage['expected']}\n")
                f.write(f"  Found: {coverage['found']}\n")
                f.write(f"  Coverage: {coverage['coverage']:.1%}\n")
                if coverage['missing']:
                    f.write(f"  Missing: {', '.join(coverage['missing'])}\n")
                f.write("\n")
        
        print(f"📊 Saved statistics: {stats_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Build GRID Corpus Manifest')
    parser.add_argument('--grid-dir', default='data/grid',
                       help='Path to GRID corpus directory')
    parser.add_argument('--output-dir', default='manifests',
                       help='Output directory for manifest files')
    parser.add_argument('--no-word-level', action='store_true',
                       help='Skip word-level manifest generation')
    
    args = parser.parse_args()
    
    print("🎯 GRID CORPUS MANIFEST BUILDER")
    print("=" * 40)
    
    # Check GRID directory
    grid_dir = Path(args.grid_dir)
    if not grid_dir.exists():
        print(f"❌ GRID directory not found: {grid_dir}")
        print("Please ensure GRID corpus is available or update --grid-dir path")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build manifest
    builder = GRIDManifestBuilder(grid_dir)
    manifest_data = builder.scan_grid_directory()
    
    if not manifest_data:
        print("❌ No GRID files found")
        return 1
    
    # Save manifest
    builder.save_manifest(output_dir, include_word_level=not args.no_word_level)
    
    print(f"\n✅ GRID manifest building complete")
    print(f"📁 Output directory: {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
