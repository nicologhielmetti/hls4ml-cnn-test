import unittest
import re
import numpy as np


def parse_sim_log(path, backend):
    cosim_file = open(path, 'r')
    vector = []
    if backend == 'VivadoAccelerator':
        for line in cosim_file.readlines():
            if line == ' \n':
                continue
            str_val = line.split(',')[0]
            value = float(re.sub("[^\d.-]", "", str_val))
            vector.append(value)
        return vector
    elif backend == 'Vivado':
        for line in cosim_file.readlines():
            for v in line.split(' '):
                if v == '\n':
                    continue
                vector.append(float(v))
        return vector
    else:
        raise Exception('Error on backend specification')


class SimValuesTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_master_cosim = np.array(parse_sim_log('V@master/V@master-cosim.log', 'Vivado'))
        self.v_master_csim = np.array(parse_sim_log('V@master/V@master-csim.log', 'Vivado'))

        self.v_pr_cosim = np.array(parse_sim_log('V@PR/V@PR-cosim.log', 'Vivado'))
        self.v_pr_csim = np.array(parse_sim_log('V@PR/V@PR-csim.log', 'Vivado'))

        self.va_cosim = np.array(parse_sim_log('VA/VA-cosim.log', 'VivadoAccelerator'))
        self.va_csim = np.array(parse_sim_log('VA/VA-csim.log', 'VivadoAccelerator'))
        self.va_hw = np.load('VA/VA-HW.npy').ravel()

        self.v_pr_dummy_csim = np.array(parse_sim_log('V@PR-dummy/V@PR-dummy-csim.log', 'Vivado'))
        self.v_pr_dummy_cosim = np.array(parse_sim_log('V@PR-dummy/V@PR-dummy-cosim.log', 'Vivado'))

        self.va_dummy_csim = np.array(parse_sim_log('VA-dummy/VA-dummy-csim.log', 'VivadoAccelerator'))
        self.va_dummy_cosim = np.array(parse_sim_log('VA-dummy/VA-dummy-cosim.log', 'VivadoAccelerator'))

        self.v_378_csim = np.array(parse_sim_log('V@378/V@378-csim.log', 'Vivado'))
        self.v_378_cosim = np.array(parse_sim_log('V@378/V@378-cosim.log', 'Vivado'))

    # PR378 test model. All layers before flatten (excluded)
    def test_v_378_csim_VS_v_378_cosim(self):
        print('v_378_csim vs v_378_cosim')
        np.testing.assert_array_equal(x=self.v_378_csim, y=self.v_378_cosim)


    # Dummy model tests
    # --- 1st row ---

    def test_va_dummy_csim_VS_va_dummy_cosim(self):
        print('va_dummy_csim vs va_dummy_cosim')
        np.testing.assert_array_equal(x=self.va_dummy_csim, y=self.va_dummy_cosim)

    def test_va_dummy_csim_VS_v_pr_dummy_csim(self):
        print('va_dummy_csim vs v_pr_dummy_csim')
        np.testing.assert_array_equal(x=self.va_dummy_csim, y=self.v_pr_dummy_csim)

    def test_va_dummy_csim_VS_v_pr_dummy_cosim(self):
        print('va_dummy_csim vs v_pr_dummy_cosim')
        np.testing.assert_array_equal(x=self.va_dummy_csim, y=self.v_pr_dummy_cosim)

    # --- 2nd row ---
    def test_va_dummy_cosim_VS_v_pr_dummy_csim(self):
        print('va_dummy_cosim vs v_pr_dummy_csim')
        np.testing.assert_array_equal(x=self.va_dummy_cosim, y=self.v_pr_dummy_csim)

    def test_va_dummy_cosim_VS_v_pr_dummy_cosim(self):
        print('va_dummy_cosim vs v_pr_dummy_cosim')
        np.testing.assert_array_equal(x=self.va_dummy_cosim, y=self.v_pr_dummy_cosim)

    # --- 3rd row ---
    def test_v_pr_dummy_csim_VS_v_pr_dummy_cosim(self):
        print('v_pr_dummy_csim vs v_pr_dummy_cosim')
        np.testing.assert_array_equal(x=self.v_pr_dummy_csim, y=self.v_pr_dummy_cosim)

    # Big model tests
    # --- 1st row ---

    def test_va_csim_VS_va_cosim(self):
        print('va_csim vs va_cosim')
        np.testing.assert_array_equal(x=self.va_csim, y=self.va_cosim)

    def test_va_csim_VS_va_hw(self):
        print('va_csim vs va_hw')
        np.testing.assert_array_equal(x=self.va_csim, y=self.va_hw)

    def test_va_csim_VS_v_pr_csim(self):
        print('va_csim vs v_pr_csim')
        np.testing.assert_array_equal(x=self.va_csim, y=self.v_pr_csim)

    def test_va_csim_VS_v_pr_cosim(self):
        print('va_csim vs v_pr_cosim')
        np.testing.assert_array_equal(x=self.va_csim, y=self.v_pr_cosim)

    def test_va_csim_VS_v_master_csim(self):
        print('va_csim vs v_master_csim')
        np.testing.assert_array_equal(x=self.va_csim, y=self.v_master_csim)

    def test_va_csim_VS_v_master_cosim(self):
        print('va_csim vs v_master_cosim')
        np.testing.assert_array_equal(x=self.va_csim, y=self.v_master_cosim)

    # --- 2nd row ---
    def test_va_cosim_VS_va_hw(self):
        print('va_cosim vs va_hw')
        np.testing.assert_array_equal(x=self.va_cosim, y=self.va_hw)

    def test_va_cosim_VS_v_pr_csim(self):
        print('va_cosim vs v_pr_csim')
        np.testing.assert_array_equal(x=self.va_cosim, y=self.v_pr_csim)

    def test_va_cosim_VS_v_pr_cosim(self):
        print('va_cosim vs v_pr_cosim')
        np.testing.assert_array_equal(x=self.va_cosim, y=self.v_pr_cosim)

    def test_va_cosim_VS_v_master_csim(self):
        print('va_cosim vs v_master_csim')
        np.testing.assert_array_equal(x=self.va_cosim, y=self.v_master_csim)

    def test_va_cosim_VS_v_master_cosim(self):
        print(' va_cosim vs v_master_cosim')
        np.testing.assert_array_equal(x=self.va_cosim, y=self.v_master_cosim)

    # --- 3rd row ---
    def test_va_hw_VS_v_pr_csim(self):
        print('va_hw vs v_pr_csim')
        np.testing.assert_array_equal(x=self.va_hw, y=self.v_pr_csim)

    def test_va_hw_VS_v_pr_cosim(self):
        print('va_hw vs v_pr_cosim')
        np.testing.assert_array_equal(x=self.va_hw, y=self.v_pr_cosim)

    def test_va_hw_VS_v_master_csim(self):
        print('va_hw vs v_master_csim')
        np.testing.assert_array_equal(x=self.va_hw, y=self.v_master_csim)

    def test_va_hw_VS_v_master_cosim(self):
        print('va_hw vs v_master_cosim')
        np.testing.assert_array_equal(x=self.va_hw, y=self.v_master_cosim)

    # --- 4th row ---
    def test_v_pr_csim_VS_v_pr_cosim(self):
        print('v_pr_csim vs v_pr_cosim')
        np.testing.assert_array_equal(x=self.v_pr_csim, y=self.v_pr_cosim)

    def test_v_pr_csim_VS_v_master_csim(self):
        print(' v_pr_csim vs v_master_csim')
        np.testing.assert_array_equal(x=self.v_pr_csim, y=self.v_master_csim)

    def test_v_pr_csim_VS_v_master_cosim(self):
        print(' v_pr_csim vs v_master_cosim')
        np.testing.assert_array_equal(x=self.v_pr_csim, y=self.v_master_cosim)

    # --- 5th row ---
    def test_v_pr_cosim_VS_v_master_csim(self):
        print('v_pr_cosim vs v_master_csim')
        np.testing.assert_array_equal(x=self.v_pr_cosim, y=self.v_master_csim)

    def test_v_pr_cosim_VS_v_master_cosim(self):
        print('v_pr_cosim vs v_master_cosim')
        np.testing.assert_array_equal(x=self.v_pr_cosim, y=self.v_master_cosim)

    # --- 6th row ---
    def test_v_master_csim_VS_v_master_cosim(self):
        print('v_master_csim vs v_master_cosim')
        np.testing.assert_array_equal(x=self.v_master_csim, y=self.v_master_cosim)


if __name__ == '__main__':
    unittest.main()
