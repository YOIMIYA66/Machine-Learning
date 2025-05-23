#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
启航者 AI - 系统测试脚本
测试前后端核心功能，确保系统正常运行
"""

import os
import sys
import json
import time
import requests
import threading
from typing import Dict, List, Any
from datetime import datetime

def print_banner():
    """打印测试横幅"""
    print("🧪 启航者 AI - 系统功能测试")
    print("=" * 60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version}")
    print("=" * 60)

def test_imports():
    """测试关键模块导入"""
    print("\n📦 模块导入测试...")
    
    modules_to_test = [
        ('config', '配置模块'),
        ('rag_core', 'RAG核心模块'),
        ('rag_core_enhanced', '增强RAG模块'),
        ('ml_agents', 'ML代理模块'),
        ('ml_agents_enhanced', '增强ML代理模块'),
        ('learning_planner', '学习路径规划模块'),
        ('tech_lab', '技术实验室模块'),
        ('app', 'Flask应用模块')
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {description}")
            passed += 1
        except ImportError as e:
            print(f"❌ {description}: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️ {description}: {e}")
            failed += 1
    
    print(f"\n导入测试结果: {passed}个成功, {failed}个失败")
    return failed == 0

def test_config():
    """测试配置文件"""
    print("\n⚙️ 配置测试...")
    
    try:
        from config import AI_STUDIO_API_KEY, KNOWLEDGE_BASE_DIR, BAIDU_LLM_MODEL_NAME
        
        tests = [
            (AI_STUDIO_API_KEY, "AI Studio API密钥"),
            (KNOWLEDGE_BASE_DIR, "知识库目录"),
            (BAIDU_LLM_MODEL_NAME, "LLM模型名称")
        ]
        
        for config_item, name in tests:
            if config_item:
                print(f"✅ {name}: 已配置")
            else:
                print(f"❌ {name}: 未配置")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def test_learning_planner():
    """测试学习路径规划功能"""
    print("\n🛤️ 学习路径规划测试...")
    
    try:
        from learning_planner import generate_learning_path, get_user_learning_paths
        
        # 测试生成学习路径
        test_path = generate_learning_path(
            user_id="test_user",
            goal="学习机器学习基础",
            prior_knowledge=[],
            weekly_hours=10
        )
        
        if test_path and test_path.get('path_id'):
            print("✅ 学习路径生成功能正常")
        else:
            print("❌ 学习路径生成失败")
            return False
        
        # 测试获取用户路径
        user_paths = get_user_learning_paths("test_user")
        if isinstance(user_paths, list):
            print("✅ 用户路径获取功能正常")
        else:
            print("❌ 用户路径获取失败")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 学习路径规划测试失败: {e}")
        return False

def test_tech_lab():
    """测试技术实验室功能"""
    print("\n🧪 技术实验室测试...")
    
    try:
        from tech_lab import get_available_models, create_experiment
        
        # 测试获取可用模型
        models = get_available_models()
        if models and len(models) > 0:
            print(f"✅ 获取到 {len(models)} 个可用模型")
        else:
            print("⚠️ 没有可用模型")
        
        # 测试创建实验
        experiment = create_experiment(
            experiment_name="测试实验",
            description="系统测试实验",
            models=["linear_regression", "decision_tree"],
            dataset="test_data"
        )
        
        if experiment and experiment.get('experiment_id'):
            print("✅ 实验创建功能正常")
        else:
            print("❌ 实验创建失败")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 技术实验室测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n📁 文件结构测试...")
    
    required_files = [
        'app.py',
        'config.py',
        'rag_core.py',
        'ml_agents.py',
        'learning_planner.py',
        'tech_lab.py',
        'templates/index.html',
        'static/js/app.js'
    ]
    
    required_dirs = [
        'templates',
        'static',
        'static/js',
        'knowledge_base',
        'data',
        'data/learning_paths',
        'data/knowledge'
    ]
    
    all_good = True
    
    # 检查文件
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件缺失")
            all_good = False
    
    # 检查目录
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - 目录缺失")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"   ✅ 已创建目录: {dir_path}/")
            except Exception as e:
                print(f"   ❌ 创建目录失败: {e}")
                all_good = False
    
    return all_good

def start_test_server():
    """启动测试服务器"""
    print("\n🌐 启动测试服务器...")
    
    try:
        from app import app, init_app
        
        # 初始化应用
        init_app()
        
        # 在单独线程中启动服务器
        def run_server():
            app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # 等待服务器启动
        time.sleep(3)
        
        # 测试服务器是否响应
        try:
            response = requests.get('http://127.0.0.1:5001/', timeout=5)
            if response.status_code == 200:
                print("✅ 测试服务器启动成功")
                return True
            else:
                print(f"❌ 服务器响应异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 服务器连接失败: {e}")
            return False
    
    except Exception as e:
        print(f"❌ 测试服务器启动失败: {e}")
        return False

def test_api_endpoints():
    """测试API端点"""
    print("\n🔌 API端点测试...")
    
    base_url = 'http://127.0.0.1:5001'
    
    # 测试主页
    try:
        response = requests.get(f'{base_url}/', timeout=5)
        if response.status_code == 200:
            print("✅ 主页端点正常")
        else:
            print(f"❌ 主页端点异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 主页端点测试失败: {e}")
        return False
    
    # 测试查询端点
    try:
        query_data = {
            "query": "什么是机器学习？",
            "mode": "general_llm"
        }
        response = requests.post(f'{base_url}/query', json=query_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if 'answer' in result:
                print("✅ 查询端点正常")
            else:
                print("❌ 查询端点响应格式异常")
                return False
        else:
            print(f"❌ 查询端点异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 查询端点测试失败: {e}")
        return False
    
    # 测试学习路径端点
    try:
        path_data = {
            "goal": "学习机器学习",
            "prior_knowledge": [],
            "weekly_hours": 10
        }
        response = requests.post(f'{base_url}/api/learning_path/create', json=path_data, timeout=10)
        if response.status_code in [200, 201]:
            print("✅ 学习路径创建端点正常")
        else:
            print(f"⚠️ 学习路径端点状态: {response.status_code}")
    except Exception as e:
        print(f"⚠️ 学习路径端点测试: {e}")
    
    return True

def test_frontend_resources():
    """测试前端资源"""
    print("\n🎨 前端资源测试...")
    
    base_url = 'http://127.0.0.1:5001'
    
    resources_to_test = [
        '/static/js/app.js',
        # 可以添加更多静态资源测试
    ]
    
    for resource in resources_to_test:
        try:
            response = requests.get(f'{base_url}{resource}', timeout=5)
            if response.status_code == 200:
                print(f"✅ {resource}")
            else:
                print(f"❌ {resource}: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ {resource}: {e}")
            return False
    
    return True

def run_comprehensive_test():
    """运行全面测试"""
    print_banner()
    
    tests = [
        ("模块导入", test_imports),
        ("配置检查", test_config),
        ("文件结构", test_file_structure),
        ("学习路径规划", test_learning_planner),
        ("技术实验室", test_tech_lab),
        ("测试服务器", start_test_server),
        ("API端点", test_api_endpoints),
        ("前端资源", test_frontend_resources)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    # 测试结果汇总
    print("\n" + "="*60)
    print("🏁 测试结果汇总")
    print("="*60)
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！系统运行正常。")
        print("您可以访问 http://localhost:5000 使用应用程序。")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} 个测试失败，请检查相关功能。")
        return False

def generate_test_report():
    """生成测试报告"""
    report_data = {
        "test_time": datetime.now().isoformat(),
        "python_version": sys.version,
        "system_info": {
            "platform": sys.platform,
            "python_executable": sys.executable
        }
    }
    
    try:
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print("\n📄 测试报告已生成: test_report.json")
    except Exception as e:
        print(f"\n⚠️ 生成测试报告失败: {e}")

def main():
    """主函数"""
    try:
        success = run_comprehensive_test()
        generate_test_report()
        
        if success:
            input("\n按回车键退出测试...")
            sys.exit(0)
        else:
            input("\n按回车键退出测试...")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生未知错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 