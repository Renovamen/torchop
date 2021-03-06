from os import path
from setuptools import setup, find_packages

current_path = path.abspath(path.dirname(__file__))

__version__ = None
ver_file = path.join(current_path, 'torchop', 'version.py')
with open(ver_file) as fp:
    exec(fp.read())

def readme():
    readme_path = path.join(current_path, 'README.md')
    with open(readme_path, encoding='utf-8') as fp:
        return fp.read()

setup(
    name = 'torchop',
    version = __version__,
    packages = find_packages(),
    description = 'A collection of some attention/convolution operators implemented using PyTorch.',
    long_description = readme(),
    long_description_content_type = 'text/markdown',
    keywords=['pytorch', 'attention', 'transformer', 'convolution', 'cnn'],
    license = 'MIT',
    author = 'Xiaohan Zou',
    author_email = 'renovamenzxh@gmail.com',
    url = 'https://github.com/Renovamen/torchop',
    install_requires = [
        'numpy>=1.14.0',
        'torch>=1.4.0',
        'tqdm',
    ]
)
