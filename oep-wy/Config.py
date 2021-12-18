import configparser


def _get_typed_arg(f, key, fallback=None, tolist=False):
    v = f(key, fallback=fallback)
    if tolist: v = v.split()
    return v


def parse_dataset_options(sec):
    dataset_opts = dict()
    dataset_opts['MeshLevel'] = _get_typed_arg(sec.getint, 'MeshLevel', fallback=3)
    dataset_opts['CubeLength'] = _get_typed_arg(sec.getfloat, 'CubeLength', fallback=0.9)
    dataset_opts['CubePoint'] = _get_typed_arg(sec.getint, 'CubePoint', fallback=9)
    dataset_opts['OutputPath'] = _get_typed_arg(sec.get, 'OutputPath', fallback='.')
    dataset_opts['OutputName'] = _get_typed_arg(sec.get, 'OutputName', fallback='data') 
    dataset_opts['Symmetric'] = _get_typed_arg(sec.get, 'Symmetric', fallback=None)
    dataset_opts['RandomTransform'] = _get_typed_arg(sec.getboolean, 'RandomTransform', fallback=False)
    return dataset_opts


def parse_oep_options(sec):
    oep_opts = dict()
    oep_opts['SpinUnrestricted'] = _get_typed_arg(sec.getboolean, 'SpinUnrestricted', fallback=False)
    oep_opts['InputDensity'] = _get_typed_arg(sec.get, 'InputDensity', tolist=True)
    oep_opts['Structure'] = _get_typed_arg(sec.get, 'Structure')
    oep_opts['OrbitalBasis'] = _get_typed_arg(sec.get, 'OrbitalBasis')
    oep_opts['PotentialBasis'] = _get_typed_arg(sec.get, 'PotentialBasis')
    oep_opts['ReferencePotential'] = _get_typed_arg(sec.get, 'ReferencePotential', fallback='hfx', tolist=True)
    oep_opts['PotentialCoefficientInit'] = _get_typed_arg(sec.get, 'PotentialCoefficientInit', fallback='zeros', tolist=True)

    oep_opts['CheckPointPath'] = _get_typed_arg(sec.get, 'CheckPointPath', fallback='.')

    oep_opts['ConvergenceCriterion'] = _get_typed_arg(sec.getfloat, 'ConvergenceCriterion', fallback=1e-8)
    oep_opts['MaxIterator'] = _get_typed_arg(sec.getint, 'MaxIterator', fallback=99)
    oep_opts['SVDCutoff'] = _get_typed_arg(sec.getfloat, 'SVDCutoff', fallback=0)
    oep_opts['LambdaRegulation'] = _get_typed_arg(sec.getfloat, 'LambdaRegulation', fallback=0)
    oep_opts['ZeroForceConstrain'] = _get_typed_arg(sec.getboolean, 'ZeroForceConstrain', fallback=False)
    oep_opts['RealSpaceAnalysis'] = _get_typed_arg(sec.getboolean, 'RealSpaceAnalysis', fallback=True)
    oep_opts['FullRealSpacePotential'] = _get_typed_arg(sec.getboolean, 'FullRealSpacePotential', fallback=False)

    return oep_opts


def get_options(config_file, sec):
    config = configparser.ConfigParser()
    config.read(config_file)
    if sec == 'OEP': 
        return parse_oep_options(config['OEP'])
    elif sec == 'DATASET': 
        return parse_dataset_options(config['DATASET'])


if __name__ == '__main__':
    oep_opts = get_options('test/oep.cfg', 'OEP')
    for k, v in oep_opts.items():
        print(k, v, v.__class__)

