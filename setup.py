from setuptools import setup, find_packages
setup(
name='tinytensor',
version='0.1.0',
author='Mihai-Alexandru Radu',
author_email='miihairadu@gmail.com',
description='An indie deep learning library.',
packages=find_packages(),
license="GPL",
include_package_data=True,
install_requires=[
        'numpy'
],
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
python_requires='>=3.6',
)