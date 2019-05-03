from setuptools import setup,find_packages
#May need to install Pystan separately with pip
setup(name='npl',
      version='0.1.0',
      description='Bayesian Nonparametric Learning',
      url='http://github.com/edfong/npl',
      author='Edwin Fong',
      author_email='edwin.fong@stats.ox.ac.uk',
      license='BSD 3-Clause',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'pandas',
          'matplotlib',
          'seaborn',
          'pystan',
          'joblib',
          'tqdm',
          'python-mnist'
      ]
      )