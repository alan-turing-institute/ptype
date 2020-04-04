from setuptools import setup

setup(
   name='ptype',
   version='1.0',
   description='Probabilistic type inference',
   author='Taha Ceritli, Christopher K. I. Williams, James Geddes',
   author_email='t.y.ceritli@sms.ed.ac.uk, ckiw@inf.ed.ac.uk, jgeddes@turing.ac.uk',
   url='https://github.com/alan-turing-institute/ptype-dmkd',
   packages=[],
   install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pandas', 'greenery', 'clevercsv']
)
