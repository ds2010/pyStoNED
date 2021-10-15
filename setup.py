from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='pystoned',
    version='0.5.4',
    description='A Python Package for Convex Regression and Frontier Estimation',
    long_description_content_type="text/markdown",
    long_description=README,
    license='GPLv3',
    packages=find_packages(),
    author='Sheng Dai, Yu-Hsueh Fang, Chia-Yen Lee, Timo Kuosmanen',
    author_email='sheng.dai@aalto.fi',
    keywords=['StoNED', 'CNLS', 'CER', 'CQR', 'Z-variables','CNLSG'],
    url='https://github.com/ds2010/pyStoNED',
    download_url='https://pypi.org/project/pystoned/',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    package_data={'pystoned': ['data/*.csv']},
)

install_requires = [
    'pyomo>=6.1.2',
    'pandas>=1.1.3',
    'numpy>=1.19.2',
    'scipy>=1.5.2',
    'matplotlib'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
