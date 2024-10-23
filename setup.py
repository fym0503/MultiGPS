import os
from setuptools import setup, find_packages

BUILD_ID = os.environ.get("BUILD_BUILDID", "0")

setup(
    name="MultiGPS",
    version="0.1" + "." + BUILD_ID,
    # Author details
    author="Yimin Fan, Yunhua Zhong",
    author_email="fanyimin@link.cuhk.edu.hk, 211300069@smail.nju.edu.cn",
    packages=["MultiGPS"],
)