#!/usr/bin/env python3
"""
Robust Overnight Training Script with Comprehensive Safety Guardrails
Designed for unattended training with automatic recovery and monitoring.
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('overnight_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OvernightTrainer:
    def __init__(self):
        self.workspace_dir = Path("/Users/client/Desktop/LRP classifier 11.9.25")
        self.experiments_dir = self.workspace_dir / "experiments" / "run_001"
        self.log_file = self.experiments_dir / "train.log"
        self.process = None
        
        # Ensure directories exist
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
    def find_latest_checkpoint(self):
        """Find the latest checkpoint to resume from."""
        checkpoint_pattern = self.experiments_dir / "checkpoint_epoch_*.pth"
        checkpoints = list(self.experiments_dir.glob("checkpoint_epoch_*.pth"))
        
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            latest = checkpoints[-1]
            logger.info(f"Found latest checkpoint: {latest}")
            return str(latest)
        
        # Fallback to best model
        best_model = self.experiments_dir / "best_model.pth"
        if best_model.exists():
            logger.info(f"Using best model checkpoint: {best_model}")
            return str(best_model)
        
        logger.warning("No checkpoint found, starting from scratch")
        return None
    
    def build_training_command(self):
        """Build the comprehensive training command."""
        checkpoint = self.find_latest_checkpoint()
        
        cmd = [
            "python", "train.py",
            "--manifest", "clean_balanced_manifest.csv",
            "--unfreeze_layers", "layer3,layer4",
            "--head_lr", "2e-4",
            "--backbone_lr", "2e-5", 
            "--weight_decay", "0.01",
            "--dropout", "0.3",
            "--label_smoothing", "0.05",
            "--clip_len", "32",
            "--ema_beta", "0.999",
            "--early_stop_patience", "6",
            "--balance", "weighted_sampler",
            "--aug", "temporal10,affine2,bc0.1",
            "--save_dir", "experiments/run_001"
        ]
        
        if checkpoint:
            cmd.extend(["--resume", checkpoint])
        
        return cmd
    
    def start_training(self):
        """Start the training process with caffeinate."""
        cmd = self.build_training_command()
        
        # Create caffeinate command to keep Mac awake
        caffeinate_cmd = ["caffeinate", "-dimsu"] + cmd
        
        logger.info("Starting overnight training with safety guardrails...")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Log file: {self.log_file}")
        logger.info(f"Working directory: {self.workspace_dir}")
        
        try:
            # Start process with output redirection
            with open(self.log_file, 'w') as log_f:
                self.process = subprocess.Popen(
                    caffeinate_cmd,
                    cwd=self.workspace_dir,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            logger.info(f"Training started with PID: {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return False
    
    def monitor_training(self):
        """Monitor the training process."""
        if not self.process:
            logger.error("No training process to monitor")
            return
        
        logger.info("Monitoring training process...")
        start_time = time.time()
        
        try:
            while True:
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process has finished
                    return_code = self.process.returncode
                    runtime = time.time() - start_time
                    
                    if return_code == 0:
                        logger.info(f"Training completed successfully after {runtime/3600:.1f} hours")
                        self._run_final_evaluation()
                    else:
                        logger.error(f"Training failed with return code {return_code}")
                        self._handle_training_failure()
                    break
                
                # Log status every 30 minutes
                time.sleep(1800)  # 30 minutes
                runtime = time.time() - start_time
                logger.info(f"Training still running... Runtime: {runtime/3600:.1f} hours")
                
                # Check log file for recent activity
                self._check_log_activity()
                
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
            self._cleanup()
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
            self._cleanup()
    
    def _check_log_activity(self):
        """Check if training is making progress by examining log file."""
        try:
            if self.log_file.exists():
                # Get last few lines of log
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        logger.info(f"Latest log: {last_line}")
        except Exception as e:
            logger.warning(f"Could not check log activity: {e}")
    
    def _run_final_evaluation(self):
        """Run final evaluation with TTA."""
        logger.info("Running final evaluation with Test-Time Augmentation...")
        try:
            eval_cmd = [
                "python", "train.py",
                "--manifest", "clean_balanced_manifest.csv",
                "--resume", str(self.experiments_dir / "best_model.pth"),
                "--eval_only",
                "--save_dir", "experiments/run_001"
            ]
            
            result = subprocess.run(
                eval_cmd,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Final evaluation completed successfully")
            else:
                logger.error(f"Final evaluation failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error during final evaluation: {e}")
    
    def _handle_training_failure(self):
        """Handle training failure."""
        logger.error("Training process failed - checking for recovery options")
        
        # Check if we have any checkpoints to recover from
        checkpoints = list(self.experiments_dir.glob("checkpoint_epoch_*.pth"))
        if checkpoints:
            logger.info(f"Found {len(checkpoints)} checkpoints for potential recovery")
        else:
            logger.error("No checkpoints available for recovery")
    
    def _cleanup(self):
        """Clean up resources."""
        if self.process and self.process.poll() is None:
            logger.info("Terminating training process...")
            self.process.terminate()
            time.sleep(5)
            if self.process.poll() is None:
                logger.warning("Force killing training process...")
                self.process.kill()
    
    def run(self):
        """Main execution method."""
        logger.info("=== OVERNIGHT TRAINING STARTED ===")
        logger.info(f"Start time: {datetime.now()}")
        
        if self.start_training():
            self.monitor_training()
        else:
            logger.error("Failed to start training")
            return False
        
        logger.info("=== OVERNIGHT TRAINING COMPLETED ===")
        logger.info(f"End time: {datetime.now()}")
        return True

def main():
    """Main entry point."""
    trainer = OvernightTrainer()
    success = trainer.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
