#!/usr/bin/env python3
"""
Fast-Track Results Tracker
==========================

Tracks and compares results from different fast-track improvement phases
to systematically measure progress toward >82% LOSO accuracy target.

Author: Augment Agent
Date: 2025-09-27
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

class FastTrackTracker:
    """Track and analyze fast-track improvement results."""
    
    def __init__(self, results_file="fast_track_results.json"):
        self.results_file = results_file
        self.results = self.load_results()
    
    def load_results(self) -> Dict:
        """Load existing results or create new tracking file."""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "baseline": {"mean_accuracy": 51.2, "std_accuracy": 6.5, "description": "Original LOSO results"},
                "experiments": {}
            }
    
    def save_results(self):
        """Save results to file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def add_experiment(self, name: str, checkpoint_dir: str, description: str = ""):
        """Add experiment results from checkpoint directory."""
        results_file = Path(checkpoint_dir) / "loso_results.json"
        
        if not results_file.exists():
            print(f"Warning: Results file not found: {results_file}")
            return
        
        with open(results_file, 'r') as f:
            loso_results = json.load(f)
        
        self.results["experiments"][name] = {
            "mean_accuracy": loso_results["mean_accuracy"] * 100,  # Convert to percentage
            "std_accuracy": loso_results["std_accuracy"] * 100,
            "individual_results": [r["best_val_accuracy"] * 100 for r in loso_results["individual_results"]],
            "checkpoint_dir": checkpoint_dir,
            "description": description
        }
        
        self.save_results()
        print(f"Added experiment: {name} - {loso_results['mean_accuracy']*100:.1f}% ± {loso_results['std_accuracy']*100:.1f}%")
    
    def compare_to_baseline(self, experiment_name: str) -> Tuple[float, str]:
        """Compare experiment to baseline and return improvement."""
        if experiment_name not in self.results["experiments"]:
            return 0.0, "Experiment not found"
        
        baseline_acc = self.results["baseline"]["mean_accuracy"]
        exp_acc = self.results["experiments"][experiment_name]["mean_accuracy"]
        
        improvement = exp_acc - baseline_acc
        
        if improvement > 5.0:
            status = "✅ SIGNIFICANT IMPROVEMENT"
        elif improvement > 0:
            status = "🔄 MINOR IMPROVEMENT"
        elif improvement > -5.0:
            status = "⚠️ SLIGHT DECLINE"
        else:
            status = "❌ SIGNIFICANT DECLINE"
        
        return improvement, status
    
    def generate_report(self) -> str:
        """Generate comprehensive progress report."""
        report = []
        report.append("🎯 FAST-TRACK PROGRESS REPORT")
        report.append("=" * 50)
        
        # Baseline
        baseline = self.results["baseline"]
        report.append(f"\n📊 BASELINE: {baseline['mean_accuracy']:.1f}% ± {baseline['std_accuracy']:.1f}%")
        report.append(f"   {baseline['description']}")
        
        # Target
        target = 82.0
        gap_to_target = target - baseline["mean_accuracy"]
        report.append(f"\n🎯 TARGET: {target:.1f}% (Gap: +{gap_to_target:.1f}%)")
        
        # Experiments
        report.append(f"\n🧪 EXPERIMENTS:")
        
        if not self.results["experiments"]:
            report.append("   No experiments recorded yet")
        else:
            # Sort by accuracy
            sorted_experiments = sorted(
                self.results["experiments"].items(),
                key=lambda x: x[1]["mean_accuracy"],
                reverse=True
            )
            
            for name, data in sorted_experiments:
                improvement, status = self.compare_to_baseline(name)
                report.append(f"\n   {name}:")
                report.append(f"     Accuracy: {data['mean_accuracy']:.1f}% ± {data['std_accuracy']:.1f}%")
                report.append(f"     vs Baseline: {improvement:+.1f}% {status}")
                report.append(f"     Description: {data['description']}")
        
        # Progress analysis
        if self.results["experiments"]:
            best_exp = max(self.results["experiments"].items(), key=lambda x: x[1]["mean_accuracy"])
            best_acc = best_exp[1]["mean_accuracy"]
            remaining_gap = target - best_acc
            
            report.append(f"\n📈 PROGRESS ANALYSIS:")
            report.append(f"   Best Result: {best_exp[0]} ({best_acc:.1f}%)")
            report.append(f"   Remaining Gap: +{remaining_gap:.1f}% to reach {target:.1f}% target")
            
            if remaining_gap <= 0:
                report.append("   🎉 TARGET ACHIEVED!")
            elif remaining_gap <= 10:
                report.append("   🔥 Close to target - final optimizations needed")
            elif remaining_gap <= 20:
                report.append("   💪 Good progress - continue with advanced techniques")
            else:
                report.append("   🚀 More improvements needed - implement next phases")
        
        return "\n".join(report)
    
    def get_phase_recommendations(self) -> List[str]:
        """Get recommendations for next phases based on current progress."""
        if not self.results["experiments"]:
            return ["Run baseline experiments first"]
        
        best_acc = max(exp["mean_accuracy"] for exp in self.results["experiments"].values())
        baseline_acc = self.results["baseline"]["mean_accuracy"]
        
        recommendations = []
        
        if best_acc < baseline_acc + 5:
            recommendations.extend([
                "🔧 Phase 3.1: Optimize cosine head hyperparameters (margin, temperature)",
                "📊 Phase 4: Add MixUp and temporal jitter augmentation",
                "🎯 Focus on architecture improvements before advanced techniques"
            ])
        elif best_acc < 65:
            recommendations.extend([
                "🚀 Phase 5: Implement GRID pretraining",
                "🔄 Phase 6: Add domain adversarial training",
                "📈 Scale successful improvements to full LOSO"
            ])
        elif best_acc < 75:
            recommendations.extend([
                "🎯 Phase 7: Dataset cleaning and quality improvements",
                "🤖 Phase 8: Ensemble methods with multiple seeds",
                "⚡ Test-time augmentation for final boost"
            ])
        else:
            recommendations.extend([
                "🎉 Excellent progress! Scale to full 6-fold LOSO",
                "🔍 Fine-tune hyperparameters for final optimization",
                "📊 Validate results with comprehensive evaluation"
            ])
        
        return recommendations

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Track fast-track improvement results")
    parser.add_argument("--add", nargs=3, metavar=("NAME", "CHECKPOINT_DIR", "DESCRIPTION"),
                       help="Add experiment results")
    parser.add_argument("--report", action="store_true", help="Generate progress report")
    parser.add_argument("--recommendations", action="store_true", help="Get phase recommendations")
    
    args = parser.parse_args()
    
    tracker = FastTrackTracker()
    
    if args.add:
        name, checkpoint_dir, description = args.add
        tracker.add_experiment(name, checkpoint_dir, description)
    
    if args.report:
        print(tracker.generate_report())
    
    if args.recommendations:
        print("\n🎯 PHASE RECOMMENDATIONS:")
        for rec in tracker.get_phase_recommendations():
            print(f"   {rec}")

if __name__ == "__main__":
    main()
