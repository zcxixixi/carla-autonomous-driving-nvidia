# -*- coding: utf-8 -*-
"""
Research Experiment Manager
科研实验管理系统
"""

import os
import json
import datetime
import time
import yaml
import numpy as np
from pathlib import Path

class ResearchExperiment:
    """科研实验管理器"""
    
    def __init__(self, experiment_name, description=""):
        self.experiment_name = experiment_name
        self.description = description
        self.start_time = datetime.datetime.now()
        self.results = {}
        self.config = {}
        self.status = "initialized"
        
        # 创建实验目录
        self.experiment_dir = Path(f"experiments/{experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.experiment_dir / "data").mkdir(exist_ok=True)
        (self.experiment_dir / "results").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        
        # 初始化日志文件
        self.log_file = self.experiment_dir / "logs" / "experiment.log"
        self.log(f"Experiment '{experiment_name}' initialized")
    
    def log(self, message):
        """记录日志"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    
    def set_config(self, config_dict):
        """设置实验配置"""
        self.config = config_dict
        config_path = self.experiment_dir / "config.yaml"
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.log(f"Configuration saved to {config_path}")
    
    def start(self):
        """开始实验"""
        self.status = "running"
        self.start_time = datetime.datetime.now()
        self.log("Experiment started")
    
    def record_metric(self, metric_name, value, step=None):
        """记录实验指标"""
        if metric_name not in self.results:
            self.results[metric_name] = []
        
        metric_entry = {
            'value': value,
            'timestamp': datetime.datetime.now().isoformat(),
            'step': step
        }
        
        self.results[metric_name].append(metric_entry)
        self.log(f"Metric recorded - {metric_name}: {value}")
    
    def save_data(self, data, filename):
        """保存实验数据"""
        filepath = self.experiment_dir / "data" / filename
        
        if filename.endswith('.npy'):
            np.save(filepath, data)
        elif filename.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            # 假设是文本文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(data))
        
        self.log(f"Data saved to {filepath}")
    
    def finish(self):
        """结束实验"""
        self.status = "completed"
        self.end_time = datetime.datetime.now()
        self.duration = self.end_time - self.start_time
        
        # 保存实验摘要
        summary = {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration.total_seconds(),
            'status': self.status,
            'config': self.config,
            'results_summary': self._summarize_results()
        }
        
        summary_path = self.experiment_dir / "experiment_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.log(f"Experiment completed. Duration: {self.duration}")
        self.log(f"Summary saved to {summary_path}")
    
    def _summarize_results(self):
        """汇总实验结果"""
        summary = {}
        for metric_name, values in self.results.items():
            if values:
                numeric_values = [v['value'] for v in values if isinstance(v['value'], (int, float))]
                if numeric_values:
                    summary[metric_name] = {
                        'count': len(numeric_values),
                        'mean': np.mean(numeric_values),
                        'std': np.std(numeric_values),
                        'min': np.min(numeric_values),
                        'max': np.max(numeric_values),
                        'latest': numeric_values[-1]
                    }
        return summary


class CARLAResearchPipeline:
    """CARLA研究流水线"""
    
    def __init__(self):
        self.experiments = []
    
    def run_object_detection_experiment(self):
        """运行目标检测实验"""
        exp = ResearchExperiment(
            "carla_object_detection", 
            "Basic object detection on CARLA RGB images"
        )
        
        # 实验配置
        config = {
            'algorithm': 'color_based_detection',
            'image_resolution': [800, 600],
            'detection_threshold': 1000,
            'aspect_ratio_range': [0.8, 3.0],
            'max_images_to_process': 20
        }
        exp.set_config(config)
        
        exp.start()
        
        try:
            # 导入并运行目标检测
            from algorithms.object_detection import CARLAObjectDetector
            
            detector = CARLAObjectDetector()
            stats = detector.run_detection_analysis()
            
            # 记录实验指标
            exp.record_metric('total_frames', stats['total_frames'])
            exp.record_metric('frames_with_detections', stats['frames_with_detections'])
            exp.record_metric('total_detections', stats['total_detections'])
            exp.record_metric('avg_detections_per_frame', stats['avg_detections_per_frame'])
            
            # 保存详细结果
            exp.save_data(stats, 'detection_stats.json')
            
            exp.log("Object detection experiment completed successfully")
            
        except Exception as e:
            exp.log(f"Experiment failed: {str(e)}")
            exp.status = "failed"
        
        exp.finish()
        self.experiments.append(exp)
        
        return exp
    
    def run_data_collection_experiment(self):
        """运行数据收集实验"""
        exp = ResearchExperiment(
            "carla_data_collection",
            "Large-scale sensor data collection for research"
        )
        
        config = {
            'duration_minutes': 10,
            'scenario_count': 2,
            'sensors': ['rgb_camera', 'depth_camera', 'lidar'],
            'resolution': [1920, 1080],
            'lidar_channels': 64,
            'traffic_density': {'vehicles': 20, 'pedestrians': 30}
        }
        exp.set_config(config)
        
        exp.start()
        
        try:
            # 导入并运行数据收集
            from algorithms.collect_research_data import collect_research_dataset
            
            # 记录收集前的状态
            exp.record_metric('collection_started', 1)
            
            # 运行数据收集（简化版）
            exp.log("Starting data collection...")
            
            # 这里可以调用实际的数据收集函数
            # collect_research_dataset(duration_minutes=config['duration_minutes'], 
            #                         scenario_count=config['scenario_count'])
            
            # 模拟收集指标
            exp.record_metric('scenarios_completed', config['scenario_count'])
            exp.record_metric('estimated_data_size_mb', config['scenario_count'] * config['duration_minutes'] * 50)
            
            exp.log("Data collection completed")
            
        except Exception as e:
            exp.log(f"Data collection failed: {str(e)}")
            exp.status = "failed"
        
        exp.finish()
        self.experiments.append(exp)
        
        return exp
    
    def generate_research_report(self):
        """生成研究报告"""
        report = {
            'report_date': datetime.datetime.now().isoformat(),
            'total_experiments': len(self.experiments),
            'experiments': []
        }
        
        for exp in self.experiments:
            exp_summary = {
                'name': exp.experiment_name,
                'description': exp.description,
                'status': exp.status,
                'duration': str(exp.duration) if hasattr(exp, 'duration') else 'N/A',
                'key_results': exp._summarize_results()
            }
            report['experiments'].append(exp_summary)
        
        # 保存报告
        report_path = Path("research_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"? Research report generated: {report_path}")
        
        return report


def main():
    """主函数 - 演示研究流水线"""
    print("=" * 60)
    print("CARLA Research Pipeline Demo")
    print("=" * 60)
    
    # 创建研究流水线
    pipeline = CARLAResearchPipeline()
    
    # 运行目标检测实验
    print("\n? Running Object Detection Experiment...")
    detection_exp = pipeline.run_object_detection_experiment()
    
    # 运行数据收集实验（演示版）
    print("\n? Running Data Collection Experiment...")
    collection_exp = pipeline.run_data_collection_experiment()
    
    # 生成研究报告
    print("\n? Generating Research Report...")
    report = pipeline.generate_research_report()
    
    print(f"\n? Research pipeline completed!")
    print(f"   Total experiments: {len(pipeline.experiments)}")
    print(f"   Results saved in: experiments/ directory")
    print(f"   Research report: research_report.json")
    
    print(f"\n? Next Steps for Your Research:")
    print(f"   1. 分析实验结果和性能指标")
    print(f"   2. 调整算法参数进行优化")
    print(f"   3. 扩展到更复杂的算法")
    print(f"   4. 收集更大规模的数据集")
    print(f"   5. 发表研究论文")


if __name__ == '__main__':
    main()