from distutils.core import setup, Extension
import os
#os.chdir("/home/zhitov01/Downloads/iad-latest/iad-3-11-1/src")

def main():
  setup(
    name="IAD",
    description="description",
    author="Daniil",
    author_email="dz337@cam.ac.uk",
    ext_modules=[Extension("iad",
    sources = [
    "ad_bound.c",
"ad_cone.c",
"ad_doubl.c",
"ad_frsnl.c",
"ad_globl.c",
"ad_layers.c",
"ad_main.c",
"ad_matrx.c",
"ad_phase.c",
"ad_prime.c",
"ad_radau.c",
"ad_start.c",
"iad_calc.c",
"iad_find.c",
"iad_io.c",
"iad_main.c",
"iad_pub.c",
"iad_python_adapter.c",
"iad_util.c",
"mc_lost.c",
"mygetopt.c",
"nr_amoeb.c",
"nr_amotr.c",
"nr_brent.c",
"nr_gaulg.c",
"nr_hj.c",
"nr_mnbrk.c",
"nr_rtsaf.c",
"nr_util.c",
"nr_zbrak.c",
"nr_zbrent.c",
"simple_file.c",
"version.c"
    ])]
  )

if (__name__ == "__main__"):
  main()
