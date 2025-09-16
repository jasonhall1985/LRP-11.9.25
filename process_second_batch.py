#!/usr/bin/env python3
"""
Process Second Batch of Videos

Processes an additional 10 videos (2 from each class) using the same standardized 
preprocessing pipeline that was successfully applied to the first validation batch.
"""

import sys
import os
from pathlib import Path
import time

# Add the current directory to Python path for imports
sys.path.append('.')

from standardized_preprocessing_pipeline import StandardizedPreprocessingPipeline

def main():
    print("🚀 Processing Second Batch of Videos")
    print("=" * 60)
    
    # Second batch selection (different from first batch)
    # First batch was: doctor 1, doctor 5, glasses 1, glasses 3, help 1, help 4, phone 1, phone 3, pillow 1, pillow 4
    second_batch_videos = [
        "data/TRAINING SET 2.9.25/doctor 6.mp4",      # doctor class
        "data/TRAINING SET 2.9.25/doctor 8.mp4",      # doctor class
        "data/TRAINING SET 2.9.25/glasses 2.mp4",     # glasses class
        "data/TRAINING SET 2.9.25/glasses 5.mp4",     # glasses class
        "data/TRAINING SET 2.9.25/help 2.mp4",        # help class
        "data/TRAINING SET 2.9.25/help 6.mp4",        # help class
        "data/TRAINING SET 2.9.25/phone 2.mp4",       # phone class
        "data/TRAINING SET 2.9.25/phone 5.mp4",       # phone class
        "data/TRAINING SET 2.9.25/pillow 2.mp4",      # pillow class
        "data/TRAINING SET 2.9.25/pillow 6.mp4"       # pillow class
    ]
    
    print(f"📋 Selected Videos for Second Batch:")
    for i, video_path in enumerate(second_batch_videos, 1):
        video_name = Path(video_path).stem
        class_name = video_name.split()[0]
        print(f"   {i:2d}. {video_name:<15} ({class_name} class)")
    
    print(f"\n🎯 Processing Pipeline Configuration:")
    print("   • Temporal preservation with dynamic FPS calculation")
    print("   • Lip-aware cropping with generous coverage (640x432px)")
    print("   • 32-frame uniform temporal sampling")
    print("   • Enhanced grayscale normalization with CLAHE")
    print("   • Robust percentile normalization (2nd-98th percentile)")
    print("   • Gamma correction (γ=1.1) for facial detail enhancement")
    print("   • Target brightness standardization (mean ≈ 128)")
    
    # Initialize pipeline with same configuration as first batch
    output_dir = "grayscale_validation_output"
    pipeline = StandardizedPreprocessingPipeline(
        output_dir=output_dir,
        target_frames=32,
        enable_visual_outputs=True
    )
    
    print(f"\n📁 Output Directory: {output_dir}/processed_videos/")
    print("\n" + "=" * 60)
    
    # Process each video
    successful_videos = []
    failed_videos = []
    processing_stats = []
    
    start_time = time.time()
    
    for i, video_path in enumerate(second_batch_videos, 1):
        video_name = Path(video_path).stem
        
        print(f"\n🔄 Processing {i}/10: {video_name}")
        print("-" * 40)
        
        if not Path(video_path).exists():
            print(f"❌ Video not found: {video_path}")
            failed_videos.append(video_name)
            continue
        
        try:
            # Process video with the standardized pipeline
            video_start_time = time.time()
            results = pipeline.process_single_video(video_path, video_name)
            processing_time = time.time() - video_start_time
            
            if results['status'] == 'success':
                successful_videos.append(video_name)
                
                # Collect processing statistics
                stats = {
                    'video_name': video_name,
                    'class': video_name.split()[0],
                    'processing_time': processing_time,
                    'detection_success_rate': results.get('detection_success_rate', 0.0),
                    'cropped_dimensions': results.get('cropped_dimensions', 'N/A'),
                    'original_duration': results.get('original_duration', 0.0),
                    'frame_count': results.get('frame_count', 0)
                }
                processing_stats.append(stats)
                
                print(f"✅ Successfully processed {video_name}")
                print(f"   • Processing time: {processing_time:.2f}s")
                print(f"   • Detection success rate: {results.get('detection_success_rate', 0.0):.1%}")
                print(f"   • Output dimensions: {results.get('cropped_dimensions', 'N/A')}")
                
            else:
                failed_videos.append(video_name)
                print(f"❌ Failed to process {video_name}")
                
        except Exception as e:
            failed_videos.append(video_name)
            print(f"❌ Error processing {video_name}: {e}")
    
    total_time = time.time() - start_time
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 SECOND BATCH PROCESSING SUMMARY")
    print("=" * 60)
    
    print(f"✅ Successfully processed: {len(successful_videos)}/10 videos")
    print(f"❌ Failed: {len(failed_videos)}/10 videos")
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    print(f"📁 Output location: {output_dir}/processed_videos/")
    
    if successful_videos:
        print(f"\n📋 Successfully Processed Videos:")
        for video in successful_videos:
            class_name = video.split()[0]
            print(f"   • {video:<15} ({class_name} class)")
    
    if failed_videos:
        print(f"\n❌ Failed Videos:")
        for video in failed_videos:
            print(f"   • {video}")
    
    # Class distribution summary
    if processing_stats:
        class_counts = {}
        for stat in processing_stats:
            class_name = stat['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n📊 Class Distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"   • {class_name:<8}: {count} videos")
    
    print(f"\n🎯 Next Step: Run frame count verification to confirm all videos have exactly 32 frames")
    print(f"   Command: cd grayscale_validation_output/processed_videos && ls -la *_processed.mp4 | wc -l")

if __name__ == "__main__":
    main()
