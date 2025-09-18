#!/usr/bin/env python3
"""
Test Working Preprocessing on 5 Random Samples
==============================================
Test the working geometric preprocessing approach on the same 5 samples
to compare with the previous centering issues.
"""

import os
import time
import random
from pathlib import Path
from working_lip_centered_preprocessing import WorkingLipCenteredPreprocessor

def main():
    # Same 5 test samples as before
    test_samples = [
        "help__useruser01__18to39__female__asian__20250902T013654_topmid.mp4",
        "phone__useruser01__18to39__male__not_specified__20250830T021534_topmid.mp4", 
        "glasses__useruser01__65plus__female__caucasian__20250723T043330_topmid.mp4",
        "my_mouth_is_dry__useruser01__40to64__female__caucasian__20250730T055627_topmid.mp4",
        "pillow__useruser01__18to39__male__caucasian__20250827T061622_topmid.mp4"
    ]
    
    source_dir = Path("data/13.9.25top7dataset_cropped")
    output_dir = Path("data/working_5_sample_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎯 TESTING WORKING PREPROCESSING - 5 RANDOM SAMPLES")
    print("=" * 70)
    print(f"📁 Source Directory: {source_dir}")
    print(f"📁 Output Directory: {output_dir}")
    print()
    
    # Initialize preprocessor
    preprocessor = WorkingLipCenteredPreprocessor()
    
    results = []
    total_time = 0
    
    for i, sample_file in enumerate(test_samples, 1):
        input_path = source_dir / sample_file
        output_path = output_dir / f"{Path(sample_file).stem}_WORKING.mp4"
        
        # Extract info from filename
        parts = sample_file.split('__')
        class_name = parts[0] if len(parts) > 0 else "unknown"
        demographics = f"{parts[2]}, {parts[3]}, {parts[4]}" if len(parts) > 4 else "unknown"
        
        print(f"🎬 SAMPLE {i}/5: {sample_file}")
        print("-" * 50)
        print(f"   Class: {class_name}")
        print(f"   Demographics: {demographics}")
        
        if input_path.exists():
            start_time = time.time()
            
            # Process video
            frames, result = preprocessor.process_video(str(input_path))
            
            processing_time = time.time() - start_time
            total_time += processing_time
            
            if frames is not None and result:
                # Save processed video
                success = preprocessor.save_video_with_dynamic_fps(
                    frames, str(output_path), result.get('original_duration', 1.0)
                )
                
                if success:
                    print(f"   ✅ Processing: SUCCESS")
                    print(f"   ⏱️  Time: {processing_time:.2f}s")
                    print(f"   📁 Output: {output_path.name}")
                    
                    results.append({
                        'sample': i,
                        'class': class_name,
                        'demographics': demographics,
                        'status': 'SUCCESS',
                        'time': processing_time,
                        'output': output_path.name
                    })
                else:
                    print(f"   ❌ Processing: FAILED (save error)")
                    results.append({
                        'sample': i,
                        'class': class_name,
                        'demographics': demographics,
                        'status': 'SAVE_FAILED',
                        'time': processing_time,
                        'output': None
                    })
            else:
                print(f"   ❌ Processing: FAILED")
                print(f"   🔍 Error: {result.get('error', 'Unknown error')}")
                results.append({
                    'sample': i,
                    'class': class_name,
                    'demographics': demographics,
                    'status': 'PROCESSING_FAILED',
                    'time': processing_time,
                    'output': None
                })
        else:
            print(f"   ❌ File not found: {input_path}")
            results.append({
                'sample': i,
                'class': class_name,
                'demographics': demographics,
                'status': 'FILE_NOT_FOUND',
                'time': 0,
                'output': None
            })
        
        print()
    
    # Summary
    print("📊 SUMMARY RESULTS")
    print("=" * 70)
    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"📈 Success Rate: {successful}/5 ({successful/5*100:.1f}%)")
    print(f"⏱️  Average Processing Time: {total_time/len(results):.2f}s")
    print()
    
    print("📋 DETAILED BREAKDOWN:")
    print("-" * 70)
    for result in results:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{result['sample']}. {result['class']:<15} | {result['time']:>6.2f}s | {status_icon} {result['status']}")
    
    print()
    print("🎯 TESTING COMPLETE")
    print(f"📁 Processed videos saved to: {output_dir}")

if __name__ == "__main__":
    main()
