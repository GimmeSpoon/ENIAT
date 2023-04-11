from setuptools import setup, find_packages

setup(
    name = 'eniat',
    version = '0.1.0',
    description='Template and wrapping for python ML packages',
    long_description=r'Weclome to the Los Pollos.',
    url='https://github.com/GimmeSpoon/ENIAT-python-machine-learning-tool',
    author='GimmeSpoon',
    license='BSD',
    classifiers='Not completed.',
    keywords='pytorch, torch, sklearn, scikit-learn, lightning, pytorch-lightning',
    packages=find_packages(include=['eniat', 'eniat.*']),
    py_modules=[],
    #python_requires='>=3.8*',
    package_data={},
    data_files=[('config', ['eniat.yaml']), ('example', ['fullconfig_example.yml'])],
    entry_points={
        'console_scripts': ['eniat=eniat.eniat:eniat']
    }
)