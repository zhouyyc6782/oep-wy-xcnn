### Folder contents
- `oep-wy`: codes for OEP and dataset generation
- `nn-train`: codes to train and test a NN model
- `xcnn`: codes to perform KS-DFT/NN using trained NN model as an xc function

### Example
An example is provided in folder `example`. 11 $\rm H_2$ and 11 $\rm HeH^+$ molecules are used.

- OEP: `python run_oep.py`
- Generate dataset: `python gen_dataset.py`
- Training: `python run_train.py`
- Testing: `python run_test.py` (only to check training progress, different from KS-SCF/NN)
- KS-SCF/NN: `python run_xcnn.py`

The model obtained from training can be used in KS-DFT/NN. A pre-trained model is also provided in `example/xcnn/saved_model/H2-HeH+_CNN_GGA_1_0.504-0.896-0.008_HFX_ll_0.9_9.dat`.


### Configuration
#### OEP and Dataset
**[OEP]**
Key | Value | Note 
----|------ | ----
InputDensity                | none | Density matrix in `ndarray` format. Will compute a CCSD 1-rdm if `none` is given.
Structure                   | structure/H2/d0500.str
OrbitalBasis                | aug-cc-pvqz
PotentialBasis              | aug-cc-pvqz
ReferencePotential          | hfx | Coulomb matrix and Hartree-Fock Exchange matrix
PotentialCoefficientInit    | zeros | Can use a txt or `ndarray` file
CheckPointPath              | oep-wy/chk/H2/d0500
ConvergenceCriterion        | 1.e-12 | Stop criterion of Newton optimization procedure. 
SVDCutoff                   | 5.e-6 | Cutoff for truncated SVD
LambdaRegulation            | 0 | Lambda value for regulation to get smooth potential. Used for multiple electrons system.
ZeroForceConstrain          | false | It seems not a good choice to use zero force constrain during optimization
RealSpaceAnalysis           | true | Output density difference between input and output density in real space

**[DATASET]**
Key | Value | Note 
----|------ | ----
MeshLevel                   | 3 
CubeLength                  | 0.9 | in Bohr
CubePoint                   | 9 | number of discrete points
OutputPath                  | oep-wy/dataset/H2
OutputName                  | d0500
Symmetric                   | xz | Transform $(x, y, z)$ to $(\sqrt{x^2 + y^2}, 0, z)$ and keep only unique points

#### Training and Testing
##### Training
**[OPTIONS]**
Key | Value | Note 
----|------ | ----
prefix | nn-train
log_path | %(prefix)s/train/train.log
verbose | False 
data_path | %(prefix)s/dataset/H2-HeH+_0.9_9.npy
model | CNN_GGA_1_zsym | The models with and without `_zsym` suffix have same architecture and only differ in output. See `nn-train/model.py` and `nn-train/const_list.py`.
model_save_path | %(prefix)s/train/model_chk/H2-HeH+_0.9_0_CNN_GGA_1.dat 
batch_size | 200
max_epoch | 200000
learning_rate | 5e-3
loss_function | MSELoss_zsym
optimiser | SGD
train_set_size | 78800
validate_set_size | 19600
enable_cuda | True
constrain | zsym | Needs to be `zsym` to use model and loss function with `_zsym` suffix

##### Testing
**[OPTIONS]**
Key | Value | Note 
----|------ | ----
prefix | nn-train
log_path | %(prefix)s/test/H2/d0500/test.log
verbose | False
data_path | %(prefix)s/dataset/H2/d0500.npy
model | CNN_GGA_1
restart | %(prefix)s/train/model_chk/H2-HeH+_0.9_0_CNN_GGA_1.dat.restart10000
batch_size | 1
loss_function | MSELoss
optimiser | SGD
test_set_size | 4920
enable_cuda | True
output_path | %(prefix)s/test/H2/d0500
constrain | none

#### KS-SCF/NN
**[XCNN]**
Key | Value | Note 
----|------ | ----
Verbose                 |  True
CheckPointPath          |  xcnn/chk/H2/d0500
EnableCuda              |  True
Structure               |  structure/H2/d0500.str
OrbitalBasis            |  aug-cc-pVQZ
ReferencePotential      |  hfx | Should be same as the one used in OEP
Model                   |  cnn_gga_1 
ModelPath               |  xcnn/saved_model/H2-HeH+_0.9_0_CNN_GGA_1.dat.restart10000
MeshLevel               |  3 | Should be same as the one used in training
CubeLength              |  0.9 | Should be same as the one used in training
CubePoint               |  9 | Should be same as the one used in training
Symmetric               |  xz+ | Similar to xz but keep only $z>0$ part. Only for $\rm H_2$
InitDensityMatrix       |  rks | Used in combination with next row to setup initial density matrix for KS-SCF/NN
xcFunctional            |  b3lypg | Follow PySCF's convention. In PySCF, `b3lyp` is different from `b3lypg` and the latter refers to conventional `B3LYP` functional.
ConvergenceCriterion    |  1.e-6 | For SCF procedure
MaxIteration            |  99
ZeroForceConstrain      |  True | Typically enabled to keep zero force condition.

## Dependencies
- numpy
- scipy
- tqdm
- ConfigParser/configparser
- PyTorch with CUDA support
- PySCF > 1.5

### Note on PySCF
A customised version of libcint is used to support extra Gaussian integrals. Therefore PySCF installed using pip/conda/docker will fail and you may have to compile it from source code. A straight workaround is described below (maybe not that efficient):
1. Download [PySCF](https://github.com/pyscf/pyscf) source code and follow its procedure to compile core module.
2. Go to `pyscf/lib/build/deps/src/libcint`, where the source code of libcint is placed.
3. Open `scripts/auto_intor.cl` and add the following two lines to the last `gen-cint` block:
```
  '("int3c1e_ovlp"              ( \, \, ))
  '("int3c1e_ipovlp"            (nabla \, \, ))
```
4. Follow the instructions at `Generating integrals` in `README` to generate new codes and place them accordingly. I choose to NOT update libcint here.
5. Go back to `pyscf/lib/build` where the command to compile PySCF core module is executed. Run `make` again and the libcint library will be updated.
6. Open `pyscf/gto/moleintor.py` and add the following two lines to `_INTOR_FUNCTIONS`
```
    'int3c1e_ovlp'              : (1, 1),
    'int3c1e_ipovlp'            : (3, 3),
```
7. Done
