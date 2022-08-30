from distutils.core import setup, Extension


def main():
  setup(
    name="Fastcount",
    description="Fastcount module in python",
    author="Mike",
    author_email="mikehuls42@gmail.com",
    ext_modules=[Extension("Fastcount", 
  #  include_dirs = ['/home/zhitov01/Downloads/iad-latest/iad-3-11-1/src'],
    sources = ["ad_frsnl.c", "simple_file.c"])]
  )

if (__name__ == "__main__"):
  main()
