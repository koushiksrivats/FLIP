class DefaultConfigs_C(object):
  src1_data = 'oulu'
  src1_train_num_frames = 1
  src2_data = 'replay'
  src2_train_num_frames = 1
  src3_data = 'msu'
  src3_train_num_frames = 1
  src4_data = 'celeb'
  src4_train_num_frames = 1
  src5_data = 'casia'
  src5_train_num_frames = 1
  tgt_data = 'casia'
  tgt_test_num_frames = 2
  weight = 0.03  # You may use 0.01 or 0.03
  gamma = 0.3
  beta = 0.5


class DefaultConfigs_I(object):
  src1_data = 'oulu'
  src1_train_num_frames = 1
  src2_data = 'casia'
  src2_train_num_frames = 1
  src3_data = 'msu'
  src3_train_num_frames = 1
  src4_data = 'celeb'
  src4_train_num_frames = 1
  src5_data = 'replay'
  src5_train_num_frames = 1
  tgt_data = 'replay'
  tgt_test_num_frames = 2
  weight = 0.03  # You may use 0.01 or 0.03
  gamma = 0.3
  beta = 0.5


class DefaultConfigs_M(object):
  src1_data = 'oulu'
  src1_train_num_frames = 1
  src2_data = 'casia'
  src2_train_num_frames = 1
  src3_data = 'replay'
  src3_train_num_frames = 1
  src4_data = 'celeb'
  src4_train_num_frames = 1
  src5_data = 'msu'
  src5_train_num_frames = 1
  tgt_data = 'msu'
  tgt_test_num_frames = 2
  weight = 1
  gamma = 0.3
  beta = 0.5


class DefaultConfigs_O(object):
  src1_data = 'replay'
  src1_train_num_frames = 1
  src2_data = 'casia'
  src2_train_num_frames = 1
  src3_data = 'msu'
  src3_train_num_frames = 1
  src4_data = 'celeb'
  src4_train_num_frames = 1
  src5_data = 'oulu'
  src5_train_num_frames = 1
  tgt_data = 'oulu'
  tgt_test_num_frames = 2
  weight = 1
  gamma = 0.3
  beta = 0.5


class DefaultConfigs_cefa(object):
  src1_data = 'wmca'
  src1_train_num_frames = 1
  src2_data = 'surf'
  src2_train_num_frames = 1
  src3_data = 'celeb'
  src3_train_num_frames = 1
  src4_data = 'celeb'
  src4_train_num_frames = 1
  src5_data = 'cefa'
  src5_train_num_frames = 1
  tgt_data = 'cefa'
  tgt_test_num_frames = 2
  weight = 1
  gamma = 0.1
  beta = 0.1


class DefaultConfigs_surf(object):
  src1_data = 'wmca'
  src1_train_num_frames = 1
  src2_data = 'cefa'
  src2_train_num_frames = 1
  src3_data = 'celeb'
  src3_train_num_frames = 1
  src4_data = 'celeb'
  src4_train_num_frames = 1
  src5_data = 'surf'
  src5_train_num_frames = 1
  tgt_data = 'surf'
  tgt_test_num_frames = 2
  weight = 0.1
  gamma = 0.1
  beta = 0.1


class DefaultConfigs_wmca(object):
  src1_data = 'cefa'
  src1_train_num_frames = 1
  src2_data = 'surf'
  src2_train_num_frames = 1
  src3_data = 'celeb'
  src3_train_num_frames = 1
  src4_data = 'celeb'
  src4_train_num_frames = 1
  src5_data = 'wmca'
  src5_train_num_frames = 1
  tgt_data = 'wmca'
  tgt_test_num_frames = 2
  weight = 0.01
  gamma = 0.1
  beta = 0.1


class DefaultConfigs_CI(object):
  src1_data = 'casia'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'replay'
  src3_train_num_frames = 1
  tgt_data = 'replay'
  tgt_test_num_frames = 2
  weight = 0.03  # You may use 0.01 or 0.03
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_CM(object):
  src1_data = 'casia'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'msu'
  src3_train_num_frames = 1
  tgt_data = 'msu'
  tgt_test_num_frames = 2
  weight = 1
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_CO(object):
  src1_data = 'casia'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'oulu'
  src3_train_num_frames = 1
  tgt_data = 'oulu'
  tgt_test_num_frames = 2
  weight = 1
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_IC(object):
  src1_data = 'replay'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'casia'
  src3_train_num_frames = 1
  tgt_data = 'casia'
  tgt_test_num_frames = 2
  weight = 0.03  # You may use 0.01 or 0.03
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_IM(object):
  src1_data = 'replay'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'msu'
  src3_train_num_frames = 1
  tgt_data = 'msu'
  tgt_test_num_frames = 2
  weight = 1
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_IO(object):
  src1_data = 'replay'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'oulu'
  src3_train_num_frames = 1
  tgt_data = 'oulu'
  tgt_test_num_frames = 2
  weight = 1
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_MC(object):
  src1_data = 'msu'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'casia'
  src3_train_num_frames = 1
  tgt_data = 'casia'
  tgt_test_num_frames = 2
  weight = 0.03  # You may use 0.01 or 0.03
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_MI(object):
  src1_data = 'msu'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'replay'
  src3_train_num_frames = 1
  tgt_data = 'replay'
  tgt_test_num_frames = 2
  weight = 0.03  # You may use 0.01 or 0.03
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_MO(object):
  src1_data = 'msu'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'oulu'
  src3_train_num_frames = 1
  tgt_data = 'oulu'
  tgt_test_num_frames = 2
  weight = 1
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_OC(object):
  src1_data = 'oulu'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'casia'
  src3_train_num_frames = 1
  tgt_data = 'casia'
  tgt_test_num_frames = 2
  weight = 0.03  # You may use 0.01 or 0.03
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_OI(object):
  src1_data = 'oulu'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'replay'
  src3_train_num_frames = 1
  tgt_data = 'replay'
  tgt_test_num_frames = 2
  weight = 0.03  # You may use 0.01 or 0.03
  gamma = 0.3
  beta = 0.5

class DefaultConfigs_OM(object):
  src1_data = 'oulu'
  src1_train_num_frames = 1
  src2_data = 'celeb'
  src2_train_num_frames = 1
  src3_data = 'msu'
  src3_train_num_frames = 1
  tgt_data = 'msu'
  tgt_test_num_frames = 2
  weight = 1
  gamma = 0.3
  beta = 0.5




class DefaultConfigs_CELEBA(object):
  tgt_data = 'celeb'
  tgt_test_num_frames = 2
  weight = 1
  gamma = 0.3
  beta = 0.5


# 0-shot / 5-shot configs
configC = DefaultConfigs_C()
configI = DefaultConfigs_I()
configM = DefaultConfigs_M()
configO = DefaultConfigs_O()

config_cefa = DefaultConfigs_cefa()
config_surf = DefaultConfigs_surf()
config_wmca = DefaultConfigs_wmca()

# 1-1 configs
config_CI = DefaultConfigs_CI
config_CM = DefaultConfigs_CM
config_CO = DefaultConfigs_CO

config_IC = DefaultConfigs_IC
config_IM = DefaultConfigs_IM
config_IO = DefaultConfigs_IO

config_MC = DefaultConfigs_MC
config_MI = DefaultConfigs_MI
config_MO = DefaultConfigs_MO

config_OC = DefaultConfigs_OC
config_OI = DefaultConfigs_OI
config_OM = DefaultConfigs_OM

# additional data configs
config_celeba = DefaultConfigs_CELEBA()