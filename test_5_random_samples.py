#!/usr/bin/env python3
"""
Test 5 Random Samples - Centering Accuracy Evaluation
Test the reverted preprocessing pipeline on 5 diverse video samples
"""

import sys
import time
from pathlib import Path
from final_corrected_preprocessing import FinalCorrectedPreprocessor

def test_5_random_samples():
    """Test preprocessing on 5 diverse video samples."""
    
    print("🎯 TESTING 5 RANDOM SAMPLES - CENTERING ACCURACY EVALUATION")
    print("=" * 70)
    
    # Select 5 diverse samples covering different classes and demographics
    test_samples = [
        "help__useruser01__18to39__female__asian__20250902T013654_topmid.mp4",  # help, young, female, asian
        "phone__useruser01__18to39__male__not_specified__20250830T021534_topmid.mp4",  # phone, young, male
        "glasses__useruser01__65plus__female__caucasian__20250723T043330_topmid.mp4",  # glasses, elderly, female
        "my_mouth_is_dry__useruser01__40to64__female__caucasian__20250730T055627_topmid.mp4",  # my_mouth_is_dry, middle-aged
        "pillow__useruser01__18to39__male__caucasian__20250827T061622_topmid.mp4"  # pillow, young, male, caucasian
    ]
    
    source_dir = Path("data/13.9.25top7dataset_cropped")
    output_dir = Path("data/5_sample_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = FinalCorrectedPreprocessor()
    
    # Test results storage
    all_results = []
    
    print(f"📁 Source Directory: {source_dir}")
    print(f"📁 Output Directory: {output_dir}")
    print()
    
    for i, sample_name in enumerate(test_samples, 1):
        print(f"🎬 SAMPLE {i}/5: {sample_name}")
        print("-" * 50)
        
        # Extract info from filename
        parts = sample_name.split("__")
        word_class = parts[0]
        demographics = f"{parts[2]}, {parts[3]}, {parts[4]}"
        
        print(f"   Class: {word_class}")
        print(f"   Demographics: {demographics}")
        
        input_path = source_dir / sample_name
        output_path = output_dir / f"{sample_name.replace('.mp4', '_TESTED.mp4')}"
        
        if not input_path.exists():
            print(f"   ❌ Video not found: {input_path}")
            continue
        
        try:
            # Process the video
            start_time = time.time()
            frames, result = preprocessor.process_video_final_corrected(str(input_path))
            processing_time = time.time() - start_time

            # Save the processed video
            if frames is not None:
                preprocessor.save_video_with_dynamic_fps(frames, str(output_path), result.get('original_duration', 1.0))
            
            if frames is not None and result:
                # Extract quality metrics from the nested result structure
                quality_control = result.get('quality_control', {})
                centering_accuracy = result.get('centering_accuracy', {})

                perfect_count = quality_control.get('perfect_centering_count', 0)
                total_frames = quality_control.get('frames_checked', 32)
                mean_deviation = quality_control.get('mean_deviation', 0.0)
                quality_passed = quality_control.get('quality_passed', False)
                
                perfect_rate = (perfect_count / total_frames) * 100 if total_frames > 0 else 0
                
                print(f"   ✅ Processing: SUCCESS")
                print(f"   ⏱️  Time: {processing_time:.2f}s")
                print(f"   🎯 Perfect Centering: {perfect_count}/{total_frames} ({perfect_rate:.1f}%)")
                print(f"   📏 Mean Deviation: {mean_deviation:.3f}px")
                print(f"   ✅ Quality: {'PASSED' if quality_passed else 'FAILED'}")
                
                # Store results
                all_results.append({
                    'sample': sample_name,
                    'class': word_class,
                    'demographics': demographics,
                    'perfect_count': perfect_count,
                    'total_frames': total_frames,
                    'perfect_rate': perfect_rate,
                    'mean_deviation': mean_deviation,
                    'quality_passed': quality_passed,
                    'processing_time': processing_time
                })
                
            else:
                print(f"   ❌ Processing: FAILED")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print()
    
    # Summary results
    print("📊 SUMMARY RESULTS")
    print("=" * 70)
    
    if all_results:
        total_perfect = sum(r['perfect_count'] for r in all_results)
        total_frames = sum(r['total_frames'] for r in all_results)
        overall_perfect_rate = (total_perfect / total_frames) * 100 if total_frames > 0 else 0
        average_deviation = sum(r['mean_deviation'] for r in all_results) / len(all_results)
        passed_count = sum(1 for r in all_results if r['quality_passed'])
        
        print(f"📈 Overall Perfect Centering Rate: {total_perfect}/{total_frames} ({overall_perfect_rate:.1f}%)")
        print(f"📏 Average Mean Deviation: {average_deviation:.3f}px")
        print(f"✅ Quality Passed: {passed_count}/{len(all_results)} samples")
        print(f"⏱️  Average Processing Time: {sum(r['processing_time'] for r in all_results) / len(all_results):.2f}s")
        print()
        
        # Detailed breakdown
        print("📋 DETAILED BREAKDOWN:")
        print("-" * 70)
        for i, result in enumerate(all_results, 1):
            status = "✅ PASS" if result['quality_passed'] else "❌ FAIL"
            print(f"{i}. {result['class']:<15} | {result['perfect_rate']:>5.1f}% | {result['mean_deviation']:>6.3f}px | {status}")
        
        print()
        
        # Analysis
        print("🔍 ANALYSIS:")
        print("-" * 70)
        
        if overall_perfect_rate >= 80:
            print("✅ EXCELLENT: Overall centering accuracy meets target (≥80%)")
        elif overall_perfect_rate >= 60:
            print("⚠️  GOOD: Centering accuracy is reasonable but needs improvement")
        elif overall_perfect_rate >= 40:
            print("⚠️  FAIR: Centering accuracy needs significant improvement")
        else:
            print("❌ POOR: Centering accuracy requires major algorithm revision")
        
        if average_deviation <= 2.0:
            print("✅ PRECISION: Average deviation meets target (≤2.0px)")
        elif average_deviation <= 5.0:
            print("⚠️  MODERATE: Average deviation is acceptable but could be better")
        else:
            print("❌ IMPRECISE: Average deviation exceeds acceptable limits")
        
        # Class-specific analysis
        class_performance = {}
        for result in all_results:
            class_name = result['class']
            if class_name not in class_performance:
                class_performance[class_name] = []
            class_performance[class_name].append(result['perfect_rate'])
        
        print()
        print("📊 CLASS-SPECIFIC PERFORMANCE:")
        for class_name, rates in class_performance.items():
            avg_rate = sum(rates) / len(rates)
            print(f"   {class_name:<15}: {avg_rate:.1f}% average perfect centering")
    
    else:
        print("❌ No successful processing results to analyze")
    
    print()
    print("🎯 TESTING COMPLETE")
    print(f"📁 Processed videos saved to: {output_dir}")

if __name__ == "__main__":
    test_5_random_samples()
