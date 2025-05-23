#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
启航者 AI - 应用程序启动脚本
包含环境检查、依赖验证和应用程序启动
"""

import os
import sys
import subprocess
import importlib
from typing import List, Tuple

def check_python_version() -> bool:
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("   需要Python 3.8或更高版本")
        return False

def check_dependencies() -> Tuple[bool, List[str]]:
    """检查必要的依赖包"""
    required_packages = [
        'flask',
        'flask_cors',
        'pandas',
        'numpy',
        'requests',
        'python-dotenv',
        'chromadb',
        'langchain',
        'scikit-learn',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_environment_variables() -> bool:
    """检查环境变量"""
    from dotenv import load_dotenv
    load_dotenv()
    
    ai_studio_key = os.getenv("AI_STUDIO_API_KEY")
    
    if ai_studio_key:
        print(f"✅ AI_STUDIO_API_KEY: {'*' * (len(ai_studio_key) - 4)}{ai_studio_key[-4:]}")
        return True
    else:
        print("❌ AI_STUDIO_API_KEY 未设置")
        print("   请在.env文件中配置您的百度AI Studio API密钥")
        return False

def check_directories() -> bool:
    """检查必要的目录结构"""
    required_dirs = [
        'templates',
        'static',
        'static/js',
        'static/css',
        'knowledge_base',
        'data',
        'data/learning_paths',
        'data/knowledge'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - 目录不存在")
            all_exist = False
            # 尝试创建目录
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"   ✅ 已创建目录: {dir_path}/")
                all_exist = True
            except Exception as e:
                print(f"   ❌ 创建目录失败: {e}")
    
    return all_exist

def check_critical_files() -> bool:
    """检查关键文件"""
    critical_files = [
        'app.py',
        'config.py',
        'rag_core.py',
        'ml_agents.py',
        'learning_planner.py',
        'tech_lab.py',
        'templates/index.html',
        'static/js/app.js'
    ]
    
    all_exist = True
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            all_exist = False
    
    return all_exist

def install_missing_packages(packages: List[str]) -> bool:
    """安装缺失的包"""
    print("\n🔄 正在安装缺失的依赖包...")
    
    for package in packages:
        try:
            print(f"   安装 {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"   ✅ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ {package} 安装失败: {e}")
            return False
    
    return True

def run_system_checks() -> bool:
    """运行所有系统检查"""
    print("🔍 启航者 AI - 系统环境检查")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 5
    
    # 1. Python版本检查
    print("\n1. Python版本检查:")
    if check_python_version():
        checks_passed += 1
    
    # 2. 依赖包检查
    print("\n2. 依赖包检查:")
    deps_ok, missing = check_dependencies()
    if deps_ok:
        checks_passed += 1
    elif missing:
        print(f"\n缺失 {len(missing)} 个依赖包")
        install_choice = input("是否自动安装缺失的依赖包？(y/n): ").lower().strip()
        if install_choice in ['y', 'yes', '是']:
            if install_missing_packages(missing):
                print("✅ 所有依赖包安装完成")
                checks_passed += 1
    
    # 3. 环境变量检查
    print("\n3. 环境变量检查:")
    if check_environment_variables():
        checks_passed += 1
    
    # 4. 目录结构检查
    print("\n4. 目录结构检查:")
    if check_directories():
        checks_passed += 1
    
    # 5. 关键文件检查
    print("\n5. 关键文件检查:")
    if check_critical_files():
        checks_passed += 1
    
    print("\n" + "=" * 50)
    print(f"检查结果: {checks_passed}/{total_checks} 项通过")
    
    return checks_passed == total_checks

def start_application():
    """启动应用程序"""
    print("\n🚀 启动启航者 AI 应用程序...")
    
    try:
        # 导入并运行应用
        from app import app, init_app
        
        # 初始化应用
        init_app()
        
        print("✅ 应用程序初始化完成")
        print("\n🌐 应用程序将在以下地址运行:")
        print("   http://localhost:5000")
        print("   http://127.0.0.1:5000")
        print("\n按 Ctrl+C 停止服务器\n")
        
        # 启动Flask应用
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # 避免在调试模式下重复运行
        )
        
    except Exception as e:
        print(f"❌ 应用程序启动失败: {e}")
        print("\n请检查错误信息并修复问题后重试")
        return False
    
    return True

def main():
    """主函数"""
    print("🎯 启航者 AI - 智能学习导航助手")
    print("版本: 1.0.0")
    print("作者: AI Assistant")
    print()
    
    # 运行系统检查
    if not run_system_checks():
        print("\n❌ 系统检查未通过，请修复上述问题后重试")
        input("\n按回车键退出...")
        sys.exit(1)
    
    print("\n✅ 所有系统检查通过！")
    
    # 启动应用程序
    start_choice = input("\n是否现在启动应用程序？(y/n): ").lower().strip()
    if start_choice in ['y', 'yes', '是', '']:
        start_application()
    else:
        print("\n应用程序未启动。您可以稍后运行 python run_app.py 来启动。")

if __name__ == "__main__":
    main() 