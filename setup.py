from setuptools import setup, find_packages
import os

setup(
    name='ml710',  
    version='0.1.0',
    description='A short description of your project',
    author='Your Name',  
    author_email='your.email@example.com',  
    license='MIT',  
    package_dir={'': 'src'},  
    packages=find_packages(where='src'),  
    python_requires='>=3.6',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)