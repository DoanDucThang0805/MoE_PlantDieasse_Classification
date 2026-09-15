"""
Dataset Subsample Utility

Lấy một phần nhỏ dữ liệu từ dataset gốc, giữ nguyên cấu trúc thư mục.

Usage:
    python cut_dataset.py --source_dir /path/to/tomato_only --output_dir /path/to/tomato_small --percentage 20
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetCutter:
    """Lấy một phần nhỏ từ dataset gốc."""
    
    def __init__(self, source_dir: str, output_dir: str, percentage: float, seed: int = 42):
        """
        Initialize dataset cutter.
        
        Args:
            source_dir: Thư mục gốc chứa dữ liệu (tomato_only)
            output_dir: Thư mục đầu ra
            percentage: Phần trăm dữ liệu cần lấy (0-100)
            seed: Random seed để reproducibility
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.percentage = percentage
        self.seed = seed
        
        random.seed(seed)
        
        # Validate
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        if not (0 < percentage <= 100):
            raise ValueError(f"Percentage must be between 0 and 100, got {percentage}")
        
        logger.info(f"DatasetCutter initialized:")
        logger.info(f"  Source: {self.source_dir}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Percentage: {percentage}%")
        logger.info(f"  Seed: {seed}")
    
    def get_disease_classes(self) -> List[str]:
        """Lấy danh sách thư mục bệnh tật."""
        disease_dirs = [
            d for d in self.source_dir.iterdir() 
            if d.is_dir()
        ]
        disease_dirs.sort()
        return [d.name for d in disease_dirs]
    
    def get_images_from_class(self, class_dir: Path) -> List[Path]:
        """Lấy danh sách ảnh từ thư mục bệnh tật."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        images = [
            f for f in class_dir.iterdir() 
            if f.is_file() and f.suffix in valid_extensions
        ]
        return images
    
    def calculate_sample_count(self, total_count: int) -> int:
        """Tính số lượng samples cần lấy."""
        return max(1, int(total_count * self.percentage / 100))
    
    def copy_dataset(self) -> Dict[str, Dict]:
        """Sao chép subset của dataset."""
        disease_classes = self.get_disease_classes()
        
        if not disease_classes:
            raise ValueError(f"No disease classes found in {self.source_dir}")
        
        logger.info(f"Found {len(disease_classes)} disease classes:")
        for cls in disease_classes:
            logger.info(f"  - {cls}")
        
        # Tạo thư mục đầu ra
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nCreated output directory: {self.output_dir}")
        
        stats = {}
        
        # Lặp qua từng bệnh tật
        for disease_class in disease_classes:
            source_class_dir = self.source_dir / disease_class
            output_class_dir = self.output_dir / disease_class
            
            # Lấy danh sách ảnh
            images = self.get_images_from_class(source_class_dir)
            total_images = len(images)
            
            if total_images == 0:
                logger.warning(f"No images found in {disease_class}")
                continue
            
            # Tính số lượng cần lấy
            sample_count = self.calculate_sample_count(total_images)
            
            # Random sample
            selected_images = random.sample(images, sample_count)
            
            # Tạo thư mục đầu ra
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy ảnh
            for img in selected_images:
                shutil.copy2(img, output_class_dir / img.name)
            
            stats[disease_class] = {
                'original': total_images,
                'sampled': sample_count,
                'ratio': f"{(sample_count/total_images)*100:.1f}%"
            }
            
            logger.info(f"{disease_class}: {total_images} → {sample_count} ({stats[disease_class]['ratio']})")
        
        return stats
    
    def print_summary(self, stats: Dict[str, Dict]):
        """Hiển thị tóm tắt."""
        print("\n" + "="*70)
        print("DATASET SUBSET SUMMARY")
        print("="*70)
        print(f"{'Disease Class':<35} {'Original':>10} {'Sampled':>10} {'Ratio':>10}")
        print("-"*70)
        
        total_original = 0
        total_sampled = 0
        
        for disease_class, info in sorted(stats.items()):
            print(f"{disease_class:<35} {info['original']:>10} {info['sampled']:>10} {info['ratio']:>10}")
            total_original += info['original']
            total_sampled += info['sampled']
        
        print("-"*70)
        ratio = f"{(total_sampled/total_original)*100:.1f}%" if total_original > 0 else "0%"
        print(f"{'TOTAL':<35} {total_original:>10} {total_sampled:>10} {ratio:>10}")
        print("="*70 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract a subset of tomato disease dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cut_dataset.py --source_dir /data/tomato_only --output_dir /data/tomato_20pct --percentage 20
  python cut_dataset.py --source_dir /data/tomato_only --output_dir /data/tomato_50pct --percentage 50
        """
    )
    
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to source dataset directory (tomato_only)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory (will be created)"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        required=True,
        help="Percentage of data to extract (0-100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("\n" + "="*70)
        logger.info("Dataset Subset Extractor")
        logger.info("="*70 + "\n")
        
        # Create cutter
        cutter = DatasetCutter(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            percentage=args.percentage,
            seed=args.seed
        )
        
        # Copy dataset
        stats = cutter.copy_dataset()
        
        # Print summary
        cutter.print_summary(stats)
        
        logger.info("✅ Dataset subset extraction completed successfully!")
        logger.info(f"Output directory: {cutter.output_dir}\n")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
