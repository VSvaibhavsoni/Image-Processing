# Image-Processing
Open In Colab
ML Project - Image Classification Using R-CNN and pretrained model
Install Matterport Mask-RCNN
#h5py for model errors VAIBHAV
!pip3 install 'h5py==2.10.0'
Requirement already satisfied: h5py==2.10.0 in /usr/local/lib/python3.7/dist-packages (2.10.0)
Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from h5py==2.10.0) (1.16.0)
Requirement already satisfied: numpy>=1.7 in /usr/local/lib/python3.7/dist-packages (from h5py==2.10.0) (1.21.5)
%tensorflow_version 1.x
!git clone https://github.com/matterport/Mask_RCNN
%cd Mask_RCNN
fatal: destination path 'Mask_RCNN' already exists and is not an empty directory.
/content/Mask_RCNN/Mask_RCNN/Mask_RCNN
with open('mrcnn/model.py') as f:
    model_file = f.read()

with open('mrcnn/model.py', 'w') as f:
    model_file = model_file.replace("self.keras_model = self.build(mode=mode, config=config)",
                                    "self.keras_model = self.build(mode=mode, config=config)\n        self.keras_model.metrics_tensors = []")
    f.write(model_file)
!pip3 install -r requirements.txt
!python3 setup.py install
Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 1)) (1.21.5)
Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 2)) (1.4.1)
Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 3)) (7.1.2)
Requirement already satisfied: cython in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 4)) (0.29.28)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 5)) (3.2.2)
Requirement already satisfied: scikit-image in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 6)) (0.18.3)
Requirement already satisfied: tensorflow>=1.3.0 in /tensorflow-1.15.2/python3.7 (from -r requirements.txt (line 7)) (1.15.2)
Requirement already satisfied: keras>=2.0.8 in /tensorflow-1.15.2/python3.7 (from -r requirements.txt (line 8)) (2.3.1)
Requirement already satisfied: opencv-python in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 9)) (4.1.2.30)
Requirement already satisfied: h5py in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 10)) (2.10.0)
Requirement already satisfied: imgaug in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 11)) (0.2.9)
Requirement already satisfied: IPython[all] in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 12)) (5.5.0)
Requirement already satisfied: tensorboard<1.16.0,>=1.15.0 in /tensorflow-1.15.2/python3.7 (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (1.15.0)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (3.3.0)
Requirement already satisfied: keras-preprocessing>=1.0.5 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (1.1.2)
Requirement already satisfied: absl-py>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (1.0.0)
Requirement already satisfied: google-pasta>=0.1.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (0.2.0)
Requirement already satisfied: tensorflow-estimator==1.15.1 in /tensorflow-1.15.2/python3.7 (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (1.15.1)
Requirement already satisfied: protobuf>=3.6.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (3.17.3)
Requirement already satisfied: keras-applications>=1.0.8 in /tensorflow-1.15.2/python3.7 (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (1.0.8)
Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (0.37.1)
Requirement already satisfied: astor>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (0.8.1)
Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (1.16.0)
Requirement already satisfied: wrapt>=1.11.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (1.13.3)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (1.1.0)
Requirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (1.44.0)
Requirement already satisfied: gast==0.2.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow>=1.3.0->-r requirements.txt (line 7)) (0.2.2)
Requirement already satisfied: pyyaml in /usr/local/lib/python3.7/dist-packages (from keras>=2.0.8->-r requirements.txt (line 8)) (3.13)
Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow>=1.3.0->-r requirements.txt (line 7)) (57.4.0)
Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow>=1.3.0->-r requirements.txt (line 7)) (1.0.1)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow>=1.3.0->-r requirements.txt (line 7)) (3.3.6)
Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard<1.16.0,>=1.15.0->tensorflow>=1.3.0->-r requirements.txt (line 7)) (4.11.2)
Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<1.16.0,>=1.15.0->tensorflow>=1.3.0->-r requirements.txt (line 7)) (3.10.0.2)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<1.16.0,>=1.15.0->tensorflow>=1.3.0->-r requirements.txt (line 7)) (3.7.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->-r requirements.txt (line 5)) (1.3.2)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->-r requirements.txt (line 5)) (2.8.2)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->-r requirements.txt (line 5)) (3.0.7)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->-r requirements.txt (line 5)) (0.11.0)
Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image->-r requirements.txt (line 6)) (1.2.0)
Requirement already satisfied: networkx>=2.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image->-r requirements.txt (line 6)) (2.6.3)
Requirement already satisfied: imageio>=2.3.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image->-r requirements.txt (line 6)) (2.4.1)
Requirement already satisfied: tifffile>=2019.7.26 in /usr/local/lib/python3.7/dist-packages (from scikit-image->-r requirements.txt (line 6)) (2021.11.2)
Requirement already satisfied: Shapely in /usr/local/lib/python3.7/dist-packages (from imgaug->-r requirements.txt (line 11)) (1.8.1.post1)
Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (5.1.1)
Requirement already satisfied: pexpect in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (4.8.0)
Requirement already satisfied: pygments in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (2.6.1)
Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (1.0.18)
Requirement already satisfied: pickleshare in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (0.7.5)
Requirement already satisfied: decorator in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (4.4.2)
Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (0.8.1)
Requirement already satisfied: nbformat in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (5.1.3)
Requirement already satisfied: qtconsole in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (5.2.2)
Requirement already satisfied: notebook in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (5.3.1)
Requirement already satisfied: ipywidgets in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (7.6.5)
Requirement already satisfied: testpath in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (0.6.0)
Requirement already satisfied: ipykernel in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (4.10.1)
Requirement already satisfied: ipyparallel in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (8.2.0)
Requirement already satisfied: nose>=0.10.1 in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (1.3.7)
Requirement already satisfied: nbconvert in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (5.6.1)
Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (2.23.0)
Requirement already satisfied: Sphinx>=1.3 in /usr/local/lib/python3.7/dist-packages (from IPython[all]->-r requirements.txt (line 12)) (1.8.6)
Requirement already satisfied: wcwidth in /usr/local/lib/python3.7/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->IPython[all]->-r requirements.txt (line 12)) (0.2.5)
Requirement already satisfied: babel!=2.0,>=1.3 in /usr/local/lib/python3.7/dist-packages (from Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (2.9.1)
Requirement already satisfied: docutils<0.18,>=0.11 in /usr/local/lib/python3.7/dist-packages (from Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (0.17.1)
Requirement already satisfied: imagesize in /usr/local/lib/python3.7/dist-packages (from Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (1.3.0)
Requirement already satisfied: sphinxcontrib-websupport in /usr/local/lib/python3.7/dist-packages (from Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (1.2.4)
Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (21.3)
Requirement already satisfied: snowballstemmer>=1.1 in /usr/local/lib/python3.7/dist-packages (from Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (2.2.0)
Requirement already satisfied: alabaster<0.8,>=0.7 in /usr/local/lib/python3.7/dist-packages (from Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (0.7.12)
Requirement already satisfied: Jinja2>=2.3 in /usr/local/lib/python3.7/dist-packages (from Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (2.11.3)
Requirement already satisfied: pytz>=2015.7 in /usr/local/lib/python3.7/dist-packages (from babel!=2.0,>=1.3->Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (2018.9)
Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from Jinja2>=2.3->Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (2.0.1)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->IPython[all]->-r requirements.txt (line 12)) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->IPython[all]->-r requirements.txt (line 12)) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->IPython[all]->-r requirements.txt (line 12)) (2021.10.8)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->IPython[all]->-r requirements.txt (line 12)) (1.24.3)
Requirement already satisfied: tornado>=4.0 in /usr/local/lib/python3.7/dist-packages (from ipykernel->IPython[all]->-r requirements.txt (line 12)) (5.1.1)
Requirement already satisfied: jupyter-client in /usr/local/lib/python3.7/dist-packages (from ipykernel->IPython[all]->-r requirements.txt (line 12)) (5.3.5)
Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from ipyparallel->IPython[all]->-r requirements.txt (line 12)) (4.63.0)
Requirement already satisfied: entrypoints in /usr/local/lib/python3.7/dist-packages (from ipyparallel->IPython[all]->-r requirements.txt (line 12)) (0.4)
Requirement already satisfied: pyzmq>=18 in /usr/local/lib/python3.7/dist-packages (from ipyparallel->IPython[all]->-r requirements.txt (line 12)) (22.3.0)
Requirement already satisfied: psutil in /usr/local/lib/python3.7/dist-packages (from ipyparallel->IPython[all]->-r requirements.txt (line 12)) (5.4.8)
Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets->IPython[all]->-r requirements.txt (line 12)) (1.0.2)
Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets->IPython[all]->-r requirements.txt (line 12)) (0.2.0)
Requirement already satisfied: widgetsnbextension~=3.5.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets->IPython[all]->-r requirements.txt (line 12)) (3.5.2)
Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /usr/local/lib/python3.7/dist-packages (from nbformat->IPython[all]->-r requirements.txt (line 12)) (4.3.3)
Requirement already satisfied: jupyter-core in /usr/local/lib/python3.7/dist-packages (from nbformat->IPython[all]->-r requirements.txt (line 12)) (4.9.2)
Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema!=2.5.0,>=2.4->nbformat->IPython[all]->-r requirements.txt (line 12)) (5.4.0)
Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema!=2.5.0,>=2.4->nbformat->IPython[all]->-r requirements.txt (line 12)) (21.4.0)
Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema!=2.5.0,>=2.4->nbformat->IPython[all]->-r requirements.txt (line 12)) (0.18.1)
Requirement already satisfied: terminado>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from notebook->IPython[all]->-r requirements.txt (line 12)) (0.13.1)
Requirement already satisfied: Send2Trash in /usr/local/lib/python3.7/dist-packages (from notebook->IPython[all]->-r requirements.txt (line 12)) (1.8.0)
Requirement already satisfied: ptyprocess in /usr/local/lib/python3.7/dist-packages (from terminado>=0.8.1->notebook->IPython[all]->-r requirements.txt (line 12)) (0.7.0)
Requirement already satisfied: bleach in /usr/local/lib/python3.7/dist-packages (from nbconvert->IPython[all]->-r requirements.txt (line 12)) (4.1.0)
Requirement already satisfied: defusedxml in /usr/local/lib/python3.7/dist-packages (from nbconvert->IPython[all]->-r requirements.txt (line 12)) (0.7.1)
Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert->IPython[all]->-r requirements.txt (line 12)) (0.8.4)
Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert->IPython[all]->-r requirements.txt (line 12)) (1.5.0)
Requirement already satisfied: webencodings in /usr/local/lib/python3.7/dist-packages (from bleach->nbconvert->IPython[all]->-r requirements.txt (line 12)) (0.5.1)
Requirement already satisfied: qtpy in /usr/local/lib/python3.7/dist-packages (from qtconsole->IPython[all]->-r requirements.txt (line 12)) (2.0.1)
Requirement already satisfied: sphinxcontrib-serializinghtml in /usr/local/lib/python3.7/dist-packages (from sphinxcontrib-websupport->Sphinx>=1.3->IPython[all]->-r requirements.txt (line 12)) (1.1.5)
WARNING:root:Fail load requirements file, so using default ones.
/usr/local/lib/python3.7/dist-packages/setuptools/dist.py:700: UserWarning: Usage of dash-separated 'description-file' will not be supported in future versions. Please use the underscore name 'description_file' instead
  % (opt, underscore_opt))
/usr/local/lib/python3.7/dist-packages/setuptools/dist.py:700: UserWarning: Usage of dash-separated 'license-file' will not be supported in future versions. Please use the underscore name 'license_file' instead
  % (opt, underscore_opt))
/usr/local/lib/python3.7/dist-packages/setuptools/dist.py:700: UserWarning: Usage of dash-separated 'requirements-file' will not be supported in future versions. Please use the underscore name 'requirements_file' instead
  % (opt, underscore_opt))
running install
running bdist_egg
running egg_info
writing mask_rcnn.egg-info/PKG-INFO
writing dependency_links to mask_rcnn.egg-info/dependency_links.txt
writing top-level names to mask_rcnn.egg-info/top_level.txt
reading manifest template 'MANIFEST.in'
adding license file 'LICENSE'
writing manifest file 'mask_rcnn.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_py
copying mrcnn/model.py -> build/lib/mrcnn
creating build/bdist.linux-x86_64/egg
creating build/bdist.linux-x86_64/egg/mrcnn
copying build/lib/mrcnn/model.py -> build/bdist.linux-x86_64/egg/mrcnn
copying build/lib/mrcnn/utils.py -> build/bdist.linux-x86_64/egg/mrcnn
copying build/lib/mrcnn/visualize.py -> build/bdist.linux-x86_64/egg/mrcnn
copying build/lib/mrcnn/__init__.py -> build/bdist.linux-x86_64/egg/mrcnn
copying build/lib/mrcnn/parallel_model.py -> build/bdist.linux-x86_64/egg/mrcnn
copying build/lib/mrcnn/config.py -> build/bdist.linux-x86_64/egg/mrcnn
byte-compiling build/bdist.linux-x86_64/egg/mrcnn/model.py to model.cpython-37.pyc
byte-compiling build/bdist.linux-x86_64/egg/mrcnn/utils.py to utils.cpython-37.pyc
byte-compiling build/bdist.linux-x86_64/egg/mrcnn/visualize.py to visualize.cpython-37.pyc
byte-compiling build/bdist.linux-x86_64/egg/mrcnn/__init__.py to __init__.cpython-37.pyc
byte-compiling build/bdist.linux-x86_64/egg/mrcnn/parallel_model.py to parallel_model.cpython-37.pyc
byte-compiling build/bdist.linux-x86_64/egg/mrcnn/config.py to config.cpython-37.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying mask_rcnn.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying mask_rcnn.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying mask_rcnn.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying mask_rcnn.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
zip_safe flag not set; analyzing archive contents...
creating 'dist/mask_rcnn-2.1-py3.7.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing mask_rcnn-2.1-py3.7.egg
Removing /usr/local/lib/python3.7/dist-packages/mask_rcnn-2.1-py3.7.egg
Copying mask_rcnn-2.1-py3.7.egg to /usr/local/lib/python3.7/dist-packages
mask-rcnn 2.1 is already the active version in easy-install.pth

Installed /usr/local/lib/python3.7/dist-packages/mask_rcnn-2.1-py3.7.egg
Processing dependencies for mask-rcnn==2.1
Finished processing dependencies for mask-rcnn==2.1
!git clone https://github.com/cocodataset/cocoapi.git
%cd cocoapi/PythonAPI
!make
%cd ../../
fatal: destination path 'cocoapi' already exists and is not an empty directory.
/content/Mask_RCNN/Mask_RCNN/Mask_RCNN/cocoapi/PythonAPI
python setup.py build_ext --inplace
running build_ext
skipping 'pycocotools/_mask.c' Cython extension (up-to-date)
building 'pycocotools._mask' extension
creating build
creating build/common
creating build/temp.linux-x86_64-3.7
creating build/temp.linux-x86_64-3.7/pycocotools
x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fdebug-prefix-map=/build/python3.7-pX47U3/python3.7-3.7.12=. -fstack-protector-strong -Wformat -Werror=format-security -g -fdebug-prefix-map=/build/python3.7-pX47U3/python3.7-3.7.12=. -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.7/dist-packages/numpy/core/include -I../common -I/usr/include/python3.7m -c ../common/maskApi.c -o build/temp.linux-x86_64-3.7/../common/maskApi.o -Wno-cpp -Wno-unused-function -std=c99
../common/maskApi.c: In function ‘rleDecode’:
../common/maskApi.c:46:7: warning: this ‘for’ clause does not guard... [-Wmisleading-indentation]
       for( k=0; k<R[i].cnts[j]; k++ ) *(M++)=v; v=!v; }}
       ^~~
../common/maskApi.c:46:49: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘for’
       for( k=0; k<R[i].cnts[j]; k++ ) *(M++)=v; v=!v; }}
                                                 ^
../common/maskApi.c: In function ‘rleFrPoly’:
../common/maskApi.c:166:3: warning: this ‘for’ clause does not guard... [-Wmisleading-indentation]
   for(j=0; j<k; j++) x[j]=(int)(scale*xy[j*2+0]+.5); x[k]=x[0];
   ^~~
../common/maskApi.c:166:54: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘for’
   for(j=0; j<k; j++) x[j]=(int)(scale*xy[j*2+0]+.5); x[k]=x[0];
                                                      ^
../common/maskApi.c:167:3: warning: this ‘for’ clause does not guard... [-Wmisleading-indentation]
   for(j=0; j<k; j++) y[j]=(int)(scale*xy[j*2+1]+.5); y[k]=y[0];
   ^~~
../common/maskApi.c:167:54: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘for’
   for(j=0; j<k; j++) y[j]=(int)(scale*xy[j*2+1]+.5); y[k]=y[0];
                                                      ^
../common/maskApi.c: In function ‘rleToString’:
../common/maskApi.c:212:7: warning: this ‘if’ clause does not guard... [-Wmisleading-indentation]
       if(more) c |= 0x20; c+=48; s[p++]=c;
       ^~
../common/maskApi.c:212:27: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘if’
       if(more) c |= 0x20; c+=48; s[p++]=c;
                           ^
../common/maskApi.c: In function ‘rleFrString’:
../common/maskApi.c:220:3: warning: this ‘while’ clause does not guard... [-Wmisleading-indentation]
   while( s[m] ) m++; cnts=malloc(sizeof(uint)*m); m=0;
   ^~~~~
../common/maskApi.c:220:22: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘while’
   while( s[m] ) m++; cnts=malloc(sizeof(uint)*m); m=0;
                      ^~~~
../common/maskApi.c:228:5: warning: this ‘if’ clause does not guard... [-Wmisleading-indentation]
     if(m>2) x+=(long) cnts[m-2]; cnts[m++]=(uint) x;
     ^~
../common/maskApi.c:228:34: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘if’
     if(m>2) x+=(long) cnts[m-2]; cnts[m++]=(uint) x;
                                  ^~~~
../common/maskApi.c: In function ‘rleToBbox’:
../common/maskApi.c:141:31: warning: ‘xp’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       if(j%2==0) xp=x; else if(xp<x) { ys=0; ye=h-1; }
                               ^
x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fdebug-prefix-map=/build/python3.7-pX47U3/python3.7-3.7.12=. -fstack-protector-strong -Wformat -Werror=format-security -g -fdebug-prefix-map=/build/python3.7-pX47U3/python3.7-3.7.12=. -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.7/dist-packages/numpy/core/include -I../common -I/usr/include/python3.7m -c pycocotools/_mask.c -o build/temp.linux-x86_64-3.7/pycocotools/_mask.o -Wno-cpp -Wno-unused-function -std=c99
creating build/lib.linux-x86_64-3.7
creating build/lib.linux-x86_64-3.7/pycocotools
x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fdebug-prefix-map=/build/python3.7-pX47U3/python3.7-3.7.12=. -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 build/temp.linux-x86_64-3.7/../common/maskApi.o build/temp.linux-x86_64-3.7/pycocotools/_mask.o -o build/lib.linux-x86_64-3.7/pycocotools/_mask.cpython-37m-x86_64-linux-gnu.so
copying build/lib.linux-x86_64-3.7/pycocotools/_mask.cpython-37m-x86_64-linux-gnu.so -> pycocotools
rm -rf build
/content/Mask_RCNN/Mask_RCNN/Mask_RCNN
Imports
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append("samples/coco/")  # To find local version
import coco

%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = "images"
Configurations
We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the CocoConfig class in coco.py.

For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the CocoConfig class and override the attributes you need to change.

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
Configurations:
BACKBONE                       resnet101
BACKBONE_STRIDES               [4, 8, 16, 32, 64]
BATCH_SIZE                     1
BBOX_STD_DEV                   [0.1 0.1 0.2 0.2]
COMPUTE_BACKBONE_SHAPE         None
DETECTION_MAX_INSTANCES        100
DETECTION_MIN_CONFIDENCE       0.7
DETECTION_NMS_THRESHOLD        0.3
FPN_CLASSIF_FC_LAYERS_SIZE     1024
GPU_COUNT                      1
GRADIENT_CLIP_NORM             5.0
IMAGES_PER_GPU                 1
IMAGE_CHANNEL_COUNT            3
IMAGE_MAX_DIM                  1024
IMAGE_META_SIZE                93
IMAGE_MIN_DIM                  800
IMAGE_MIN_SCALE                0
IMAGE_RESIZE_MODE              square
IMAGE_SHAPE                    [1024 1024    3]
LEARNING_MOMENTUM              0.9
LEARNING_RATE                  0.001
LOSS_WEIGHTS                   {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 1.0}
MASK_POOL_SIZE                 14
MASK_SHAPE                     [28, 28]
MAX_GT_INSTANCES               100
MEAN_PIXEL                     [123.7 116.8 103.9]
MINI_MASK_SHAPE                (56, 56)
NAME                           coco
NUM_CLASSES                    81
POOL_SIZE                      7
POST_NMS_ROIS_INFERENCE        1000
POST_NMS_ROIS_TRAINING         2000
PRE_NMS_LIMIT                  6000
ROI_POSITIVE_RATIO             0.33
RPN_ANCHOR_RATIOS              [0.5, 1, 2]
RPN_ANCHOR_SCALES              (32, 64, 128, 256, 512)
RPN_ANCHOR_STRIDE              1
RPN_BBOX_STD_DEV               [0.1 0.1 0.2 0.2]
RPN_NMS_THRESHOLD              0.7
RPN_TRAIN_ANCHORS_PER_IMAGE    256
STEPS_PER_EPOCH                1000
TOP_DOWN_PYRAMID_SIZE          256
TRAIN_BN                       False
TRAIN_ROIS_PER_IMAGE           200
USE_MINI_MASK                  True
USE_RPN_ROIS                   True
VALIDATION_STEPS               50
WEIGHT_DECAY                   0.0001


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
Create Model and Load Trained Weights
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
Run Object Detection
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
Processing 1 images
image                    shape: (426, 640, 3)         min:    0.00000  max:  255.00000  uint8
molded_images            shape: (1, 1024, 1024, 3)    min: -123.70000  max:  151.10000  float64
image_metas              shape: (1, 93)               min:    0.00000  max: 1024.00000  float64
anchors                  shape: (1, 261888, 4)        min:   -0.35390  max:    1.29134  float32
