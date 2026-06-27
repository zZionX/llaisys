from setuptools import setup, find_packages

setup(
    name='nanonona', 
    version="0.1.0",
    # 自动寻找当前目录下所有包含 __init__.py 的文件夹作为包
    packages=find_packages(where=".", include=['nanonona*']), 
)