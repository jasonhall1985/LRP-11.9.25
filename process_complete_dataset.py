#!/usr/bin/env python3
"""
Process Complete Dataset

Processes ALL remaining videos from the complete dataset using the exact same 
standardized preprocessing pipeline that was successfully validated on the 20-video sample.
"""

import sys
import os
from pathlib import Path
import time
import glob

# Add the current directory to Python path for imports
sys.path.append('.')

from standardized_preprocessing_pipeline import StandardizedPreprocessingPipeline

def get_all_videos_to_process():
    """Get all videos from source directories, excluding already processed ones."""
    
    source_directories = [
        "data/TRAINING SET 2.9.25",
        "data/TEST SET", 
        "data/VAL SET"
    ]
    
    # Get all videos from source directories
    all_videos = []
    for source_dir in source_directories:
        if os.path.exists(source_dir):
            videos = glob.glob(os.path.join(source_dir, "*.mp4"))
            for video in videos:
                video_name = Path(video).stem
                all_videos.append({
                    'path': video,
                    'name': video_name,
                    'source': source_dir
                })
    
    # Get already processed videos
    output_dir = "grayscale_validation_output/processed_videos"
    processed_videos = set()
    if os.path.exists(output_dir):
        processed_files = glob.glob(os.path.join(output_dir, "*_processed.mp4"))
        for processed_file in processed_files:
            # Extract original name by removing "_processed.mp4"
            original_name = Path(processed_file).stem.replace("_processed", "")
            processed_videos.add(original_name)
    
    # Filter out already processed videos
    videos_to_process = []
    for video in all_videos:
        if video['name'] not in processed_videos:
            videos_to_process.append(video)
    
    return videos_to_process, processed_videos

def main():
    print("🚀 PROCESSING COMPLETE DATASET")
    print("=" * 80)
    
    # Get videos to process
    videos_to_process, already_processed = get_all_videos_to_process()
    
    print(f"📊 Dataset Analysis:")
    print(f"   • Already processed: {len(already_processed)} videos")
    print(f"   • Remaining to process: {len(videos_to_process)} videos")
    print(f"   • Total dataset size: {len(already_processed) + len(videos_to_process)} videos")
    
    if not videos_to_process:
        print("\n✅ All videos have already been processed!")
        return
    
    # Group by source directory for reporting
    source_counts = {}
    for video in videos_to_process:
        source = video['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n📋 Videos to Process by Source:")
    for source, count in source_counts.items():
        print(f"   • {source}: {count} videos")
    
    print(f"\n🎯 Processing Pipeline Configuration:")
    print("   • Temporal preservation with dynamic FPS calculation")
    print("   • Lip-aware cropping with generous coverage (640x432px)")
    print("   • 32-frame uniform temporal sampling")
    print("   • Enhanced grayscale normalization with CLAHE")
    print("   • Robust percentile normalization (2nd-98th percentile)")
    print("   • Gamma correction (γ=1.1) for facial detail enhancement")
    print("   • Target brightness standardization (mean ≈ 128)")
    
    # Initialize pipeline with same configuration as validated batches
    output_dir = "grayscale_validation_output"
    pipeline = StandardizedPreprocessingPipeline(
        output_dir=output_dir,
        target_frames=32,
        enable_visual_outputs=True
    )
    
    print(f"\n📁 Output Directory: {output_dir}/processed_videos/")
    print("\n" + "=" * 80)
    
    # Process each video
    successful_videos = []
    failed_videos = []
    processing_stats = []
    
    start_time = time.time()
    
    for i, video_info in enumerate(videos_to_process, 1):
        video_path = video_info['path']
        video_name = video_info['name']
        source_dir = video_info['source']
        
        print(f"\n🔄 Processing {i}/{len(videos_to_process)}: {video_name}")
        print(f"   Source: {source_dir}")
        print("-" * 60)
        
        if not Path(video_path).exists():
            print(f"❌ Video not found: {video_path}")
            failed_videos.append({
                'name': video_name,
                'source': source_dir,
                'error': 'File not found'
            })
            continue
        
        try:
            # Process video with the standardized pipeline
            video_start_time = time.time()
            results = pipeline.process_single_video(video_path, video_name)
            processing_time = time.time() - video_start_time
            
            if results['status'] == 'success':
                successful_videos.append({
                    'name': video_name,
                    'source': source_dir,
                    'processing_time': processing_time
                })
                
                # Collect processing statistics
                stats = {
                    'video_name': video_name,
                    'source': source_dir,
                    'class': video_name.split()[0] if ' ' in video_name else 'unknown',
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
                failed_videos.append({
                    'name': video_name,
                    'source': source_dir,
                    'error': results.get('error', 'Unknown error')
                })
                print(f"❌ Failed to process {video_name}")
                print(f"   • Error: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            failed_videos.append({
                'name': video_name,
                'source': source_dir,
                'error': str(e)
            })
            print(f"❌ Error processing {video_name}: {e}")
        
        # Progress update every 10 videos
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = len(videos_to_process) - i
            eta = remaining * avg_time
            print(f"\n📊 Progress Update: {i}/{len(videos_to_process)} completed")
            print(f"   • Success rate: {len(successful_videos)}/{i} ({len(successful_videos)/i*100:.1f}%)")
            print(f"   • Average processing time: {avg_time:.2f}s per video")
            print(f"   • ETA: {eta/60:.1f} minutes remaining")
    
    total_time = time.time() - start_time
    
    # Generate comprehensive summary report
    print("\n" + "=" * 80)
    print("📊 COMPLETE DATASET PROCESSING SUMMARY")
    print("=" * 80)
    
    print(f"✅ Successfully processed: {len(successful_videos)}/{len(videos_to_process)} videos")
    print(f"❌ Failed: {len(failed_videos)}/{len(videos_to_process)} videos")
    print(f"⏱️  Total processing time: {total_time/60:.2f} minutes")
    print(f"📈 Success rate: {len(successful_videos)/len(videos_to_process)*100:.1f}%")
    print(f"📁 Output location: {output_dir}/processed_videos/")
    
    # Source-wise summary
    if successful_videos:
        print(f"\n📋 Successfully Processed Videos by Source:")
        source_success = {}
        for video in successful_videos:
            source = video['source']
            source_success[source] = source_success.get(source, 0) + 1
        
        for source, count in sorted(source_success.items()):
            print(f"   • {source}: {count} videos")
    
    if failed_videos:
        print(f"\n❌ Failed Videos by Source:")
        source_failures = {}
        for video in failed_videos:
            source = video['source']
            source_failures[source] = source_failures.get(source, 0) + 1
        
        for source, count in sorted(source_failures.items()):
            print(f"   • {source}: {count} videos")
    
    # Class distribution summary (for successfully processed videos)
    if processing_stats:
        class_counts = {}
        for stat in processing_stats:
            class_name = stat['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n📊 Class Distribution (Newly Processed):")
        for class_name, count in sorted(class_counts.items()):
            print(f"   • {class_name:<12}: {count} videos")
    
    # Final dataset summary
    total_processed = len(already_processed) + len(successful_videos)
    print(f"\n🎯 FINAL DATASET STATUS:")
    print(f"   • Total processed videos: {total_processed}")
    print(f"   • Previously processed: {len(already_processed)}")
    print(f"   • Newly processed: {len(successful_videos)}")
    print(f"   • Failed processing: {len(failed_videos)}")
    
    if len(successful_videos) > 0:
        print(f"\n✅ SUCCESS! Dataset processing completed.")
        print(f"   Ready for frame count verification and model training.")
    else:
        print(f"\n⚠️  No new videos were processed successfully.")

if __name__ == "__main__":
    main()
